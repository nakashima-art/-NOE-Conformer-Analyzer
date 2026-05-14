import math
import re
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


APP_VERSION = "ver. 1.1"

R_KCAL = 0.00198720425864083  # kcal mol^-1 K^-1


# =========================
# Basic utility functions
# =========================

def convert_to_zero_based(atom_numbers, numbering_mode):
    if numbering_mode == "1-based atom numbers":
        return [x - 1 for x in atom_numbers]
    return atom_numbers


def get_mol_name(mol, index):
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    return f"conf_{index + 1}"


def mol_properties_to_dict(mol):
    props = {}
    for prop_name in mol.GetPropNames():
        try:
            props[prop_name] = mol.GetProp(prop_name)
        except Exception:
            props[prop_name] = ""
    return props


def make_atom_index_preview(mol):
    rows = []
    for atom in mol.GetAtoms():
        idx0 = atom.GetIdx()
        rows.append(
            {
                "0-based RDKit index": idx0,
                "1-based atom number": idx0 + 1,
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "degree": atom.GetDegree(),
                "formal_charge": atom.GetFormalCharge(),
            }
        )
    return pd.DataFrame(rows)


def read_sdf(uploaded_file, sanitize=False):
    sdf_bytes = uploaded_file.getvalue()

    supplier = Chem.ForwardSDMolSupplier(
        BytesIO(sdf_bytes),
        sanitize=sanitize,
        removeHs=False,
    )

    return [mol for mol in supplier if mol is not None]


def get_available_sdf_properties(mols):
    """
    Get SDF properties that are likely to contain conformer energies.
    Population-related properties are excluded.
    """
    prop_names = set()

    for mol in mols:
        for prop_name in mol.GetPropNames():
            prop_names.add(prop_name)

    prop_names = sorted(prop_names)

    # Exclude properties that are clearly not energies
    excluded_keywords = [
        "POPULATION",
        "BOLTZMANN",
        "RATIO",
        "PERCENT",
        "%",
    ]

    energy_like_keywords = [
        "ENERGY",
        "GIBBS",
        "FREE",
        "SCF",
        "POTENTIAL",
        "ELECTRONIC",
    ]

    energy_props = []

    for prop in prop_names:
        prop_upper = prop.upper()

        if any(keyword in prop_upper for keyword in excluded_keywords):
            continue

        if any(keyword in prop_upper for keyword in energy_like_keywords):
            energy_props.append(prop)

    # If no energy-like properties are found, fall back to all properties
    # except obvious population properties.
    if not energy_props:
        energy_props = [
            prop for prop in prop_names
            if not any(keyword in prop.upper() for keyword in excluded_keywords)
        ]

    return energy_props


def extract_energy_from_property(mol, property_name):
    """
    Extract numeric energy from an SDF property.
    The first numeric value is used.
    """
    if not mol.HasProp(property_name):
        return None

    value = mol.GetProp(property_name).strip()

    match = re.search(
        r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?",
        value,
    )

    if not match:
        return None

    num_text = match.group(0).replace("D", "E").replace("d", "e")

    try:
        return float(num_text)
    except Exception:
        return None


def convert_energy_to_kcal_relative(energies, energy_unit):
    """
    Convert absolute energies to relative energies in kcal/mol.
    """
    df = pd.DataFrame({"energy_raw": energies})

    if df["energy_raw"].isna().any():
        return None

    if energy_unit == "Hartree":
        factor = 627.509474
    elif energy_unit == "kcal/mol":
        factor = 1.0
    elif energy_unit == "kJ/mol":
        factor = 0.239005736
    else:
        raise ValueError("Unsupported energy unit.")

    min_energy = df["energy_raw"].min()
    rel_kcal = (df["energy_raw"] - min_energy) * factor

    return rel_kcal.tolist()


def calculate_boltzmann_populations(
    relative_energies_kcal,
    temperature,
    strictness_factor,
    energy_window,
):
    """
    Calculate Boltzmann populations.

    strictness_factor:
        1.0 = normal Boltzmann distribution
        >1.0 = softer energy penalty
    """
    weights = []

    for dE in relative_energies_kcal:
        if dE > energy_window:
            weights.append(0.0)
            continue

        effective_dE = dE / strictness_factor
        weight = math.exp(-effective_dE / (R_KCAL * temperature))
        weights.append(weight)

    total = sum(weights)

    if total <= 0:
        return [0.0 for _ in weights]

    return [w / total for w in weights]


# =========================
# NOE scoring functions
# =========================

def distance_to_score(distance):
    """
    Empirical conformer-level NOE score.
    This is not a physical probability.
    """
    if distance <= 2.5:
        return 1.0
    if distance <= 3.0:
        return 0.75
    if distance <= 3.5:
        return 0.40
    if distance <= 4.0:
        return 0.15
    return 0.0


def classify_effective_distance(r_eff):
    if r_eff <= 2.5:
        return "strong NOE likely"
    if r_eff <= 3.0:
        return "moderate NOE likely"
    if r_eff <= 3.5:
        return "weak/possible NOE"
    if r_eff <= 4.0:
        return "borderline"
    return "unlikely"


def validate_atom_group(atom_group, mol, group_name, pair_name, mol_name):
    if not atom_group:
        raise ValueError(f"{pair_name}: {group_name} is empty.")

    if any(idx < 0 or idx >= mol.GetNumAtoms() for idx in atom_group):
        raise ValueError(
            f"Invalid atom number in {pair_name}, {group_name}, "
            f"for molecule {mol_name}. This molecule has {mol.GetNumAtoms()} atoms."
        )


def calculate_noe_for_group_pairs(
    mols,
    pair_definitions,
    numbering_mode,
    populations,
    relative_energies_kcal,
    energy_values,
    include_props,
):
    """
    Calculate NOE-related quantities for proton groups.

    For each NOE definition:
        group_A_atoms = [H1, equivalent H1, ...]
        group_B_atoms = [H2, equivalent H2, ...]

    For each conformer:
        group_r6_sum = sum over all A-B combinations of r^-6

    Then:
        total weighted r^-6 = sum_i population_i * group_r6_sum_i
    """
    detailed_rows = []
    pairwise_rows = []

    for conf_idx, mol in enumerate(mols):
        if mol is None or mol.GetNumConformers() == 0:
            continue

        conf = mol.GetConformer()
        mol_name = get_mol_name(mol, conf_idx)
        props = mol_properties_to_dict(mol) if include_props else {}

        for pair in pair_definitions:
            pair_name = pair["name"]
            group_A_input = pair["group_A_atoms"]
            group_B_input = pair["group_B_atoms"]

            group_A_idx = convert_to_zero_based(group_A_input, numbering_mode)
            group_B_idx = convert_to_zero_based(group_B_input, numbering_mode)

            validate_atom_group(
                group_A_idx,
                mol,
                "Proton A group",
                pair_name,
                mol_name,
            )
            validate_atom_group(
                group_B_idx,
                mol,
                "Proton B group",
                pair_name,
                mol_name,
            )

            group_r6_sum = 0.0
            group_score_sum = 0.0
            distances = []
            pairwise_contributions = []

            for a_input, a_idx in zip(group_A_input, group_A_idx):
                for b_input, b_idx in zip(group_B_input, group_B_idx):
                    if a_idx == b_idx:
                        continue

                    distance = rdMolTransforms.GetBondLength(conf, a_idx, b_idx)

                    if distance <= 0:
                        r_minus_6 = 0.0
                    else:
                        r_minus_6 = distance ** -6

                    score = distance_to_score(distance)

                    group_r6_sum += r_minus_6
                    group_score_sum += score

                    distances.append(distance)

                    pairwise_contributions.append(
                        {
                            "atom_pair_input": f"{a_input}-{b_input}",
                            "atom_pair_0_based": f"{a_idx}-{b_idx}",
                            "distance": distance,
                            "r_minus_6": r_minus_6,
                            "score": score,
                        }
                    )

                    pairwise_row = {
                        "pair_name": pair_name,
                        "group_A_atoms": ",".join(map(str, group_A_input)),
                        "group_B_atoms": ",".join(map(str, group_B_input)),
                        "atom_pair_input": f"{a_input}-{b_input}",
                        "atom_pair_0_based": f"{a_idx}-{b_idx}",
                        "conformer_index": conf_idx + 1,
                        "conformer_name": mol_name,
                        "energy_raw": energy_values[conf_idx],
                        "relative_energy_kcal_mol": relative_energies_kcal[conf_idx],
                        "population": populations[conf_idx],
                        "population_percent": populations[conf_idx] * 100,
                        "H_H_distance_A": distance,
                        "r_minus_6": r_minus_6,
                        "population_weighted_r_minus_6": populations[conf_idx] * r_minus_6,
                        "distance_score": score,
                        "population_weighted_score": populations[conf_idx] * score,
                    }
                    pairwise_row.update(props)
                    pairwise_rows.append(pairwise_row)

            number_of_atom_pairs = len(pairwise_contributions)

            if number_of_atom_pairs == 0:
                min_distance = None
                mean_distance = None
                group_r6_average = 0.0
                group_score_average = 0.0
                conformer_effective_distance = None
                conformer_normalized_effective_distance = None
            else:
                min_distance = min(distances)
                mean_distance = sum(distances) / len(distances)

                group_r6_average = group_r6_sum / number_of_atom_pairs
                group_score_average = group_score_sum / number_of_atom_pairs

                conformer_effective_distance = (
                    group_r6_sum ** (-1 / 6) if group_r6_sum > 0 else None
                )
                conformer_normalized_effective_distance = (
                    group_r6_average ** (-1 / 6) if group_r6_average > 0 else None
                )

            row = {
                "pair_name": pair_name,
                "group_A_atoms": ",".join(map(str, group_A_input)),
                "group_B_atoms": ",".join(map(str, group_B_input)),
                "number_of_atom_pairs": number_of_atom_pairs,
                "conformer_index": conf_idx + 1,
                "conformer_name": mol_name,
                "energy_raw": energy_values[conf_idx],
                "relative_energy_kcal_mol": relative_energies_kcal[conf_idx],
                "population": populations[conf_idx],
                "population_percent": populations[conf_idx] * 100,
                "minimum_H_H_distance_A_in_group": min_distance,
                "mean_H_H_distance_A_in_group": mean_distance,
                "group_r_minus_6_sum": group_r6_sum,
                "group_r_minus_6_average": group_r6_average,
                "population_weighted_group_r_minus_6_sum": populations[conf_idx] * group_r6_sum,
                "population_weighted_group_r_minus_6_average": populations[conf_idx] * group_r6_average,
                "conformer_effective_distance_A": conformer_effective_distance,
                "conformer_normalized_effective_distance_A": conformer_normalized_effective_distance,
                "group_score_sum": group_score_sum,
                "group_score_average": group_score_average,
                "population_weighted_group_score_average": populations[conf_idx] * group_score_average,
            }

            row.update(props)
            detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    summary_rows = []

    if detailed_df.empty:
        return pd.DataFrame(), detailed_df, pairwise_df

    for pair_name, group in detailed_df.groupby("pair_name"):
        weighted_r6_sum = group["population_weighted_group_r_minus_6_sum"].sum()
        weighted_r6_average = group["population_weighted_group_r_minus_6_average"].sum()
        weighted_score_average = group["population_weighted_group_score_average"].sum()

        if weighted_r6_sum > 0:
            effective_distance = weighted_r6_sum ** (-1 / 6)
        else:
            effective_distance = None

        if weighted_r6_average > 0:
            normalized_effective_distance = weighted_r6_average ** (-1 / 6)
        else:
            normalized_effective_distance = None

        if normalized_effective_distance is not None:
            prediction = classify_effective_distance(normalized_effective_distance)
        elif effective_distance is not None:
            prediction = classify_effective_distance(effective_distance)
        else:
            prediction = "not calculated"

        likelihood_score = weighted_score_average * 100

        pairwise_group = pairwise_df[pairwise_df["pair_name"] == pair_name]

        if not pairwise_group.empty:
            closest_row = pairwise_group.loc[pairwise_group["H_H_distance_A"].idxmin()]
            largest_contribution_row = pairwise_group.loc[
                pairwise_group["population_weighted_r_minus_6"].idxmax()
            ]

            total_pairwise_weighted_r6 = pairwise_group[
                "population_weighted_r_minus_6"
            ].sum()

            if total_pairwise_weighted_r6 > 0:
                largest_contribution_percent = (
                    largest_contribution_row["population_weighted_r_minus_6"]
                    / total_pairwise_weighted_r6
                    * 100
                )
            else:
                largest_contribution_percent = None

            minimum_distance = closest_row["H_H_distance_A"]
            closest_conformer = closest_row["conformer_name"]
            closest_atom_pair = closest_row["atom_pair_input"]
            largest_contribution_conformer = largest_contribution_row["conformer_name"]
            largest_contribution_atom_pair = largest_contribution_row["atom_pair_input"]
        else:
            minimum_distance = None
            closest_conformer = None
            closest_atom_pair = None
            largest_contribution_conformer = None
            largest_contribution_atom_pair = None
            largest_contribution_percent = None

        first = group.iloc[0]

        summary_rows.append(
            {
                "pair_name": pair_name,
                "group_A_atoms": first["group_A_atoms"],
                "group_B_atoms": first["group_B_atoms"],
                "number_of_atom_pairs": int(first["number_of_atom_pairs"]),
                "effective_NOE_distance_A_total_r6": effective_distance,
                "normalized_effective_NOE_distance_A": normalized_effective_distance,
                "NOE_likelihood_score_percent": likelihood_score,
                "prediction": prediction,
                "minimum_H_H_distance_A": minimum_distance,
                "closest_atom_pair": closest_atom_pair,
                "closest_conformer": closest_conformer,
                "largest_NOE_contribution_atom_pair": largest_contribution_atom_pair,
                "largest_NOE_contribution_conformer": largest_contribution_conformer,
                "largest_NOE_contribution_percent_of_total_r6": largest_contribution_percent,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, detailed_df, pairwise_df


def make_wide_table(df, value_column):
    if df.empty:
        return pd.DataFrame()

    wide = df.pivot_table(
        index=[
            "conformer_index",
            "conformer_name",
            "relative_energy_kcal_mol",
            "population_percent",
        ],
        columns="pair_name",
        values=value_column,
        aggfunc="first",
    ).reset_index()

    wide.columns.name = None
    return wide


def convert_df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def reset_current_equivalent_fields():
    st.session_state.eq_A_count = 0
    st.session_state.eq_B_count = 0


# =========================
# Session state
# =========================

if "noe_pairs" not in st.session_state:
    st.session_state.noe_pairs = []

if "eq_A_count" not in st.session_state:
    st.session_state.eq_A_count = 0

if "eq_B_count" not in st.session_state:
    st.session_state.eq_B_count = 0


# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="NOE Conformer Analyzer",
    page_icon="🧲",
    layout="wide",
)

st.title("NOE Conformer Analyzer")
st.caption(APP_VERSION)

st.write(
    "This app estimates whether NOE correlations are likely to be observed "
    "from conformer geometries in an SDF file. It calculates H···H distances "
    "for each conformer, derives Boltzmann populations from SDF energy properties, "
    "and evaluates population-weighted r⁻⁶ effective distances. "
    "Equivalent protons can be added only when needed."
)


# =========================
# Sidebar
# =========================

st.sidebar.header("Input settings")

numbering_mode = st.sidebar.radio(
    "Atom numbering in your input",
    ["1-based atom numbers", "0-based RDKit atom indices"],
    index=0,
    help=(
        "Use 1-based atom numbers when specifying atom numbers from Gaussian, "
        "GaussView, Chem3D, Avogadro, etc."
    ),
)

sanitize = st.sidebar.checkbox(
    "Sanitize SDF while reading",
    value=False,
    help="If SDF reading fails, try turning this off.",
)

include_props = st.sidebar.checkbox(
    "Include SDF properties in detailed outputs",
    value=True,
)

show_atom_preview = st.sidebar.checkbox(
    "Show atom-index preview for the first conformer",
    value=True,
)

st.sidebar.header("Boltzmann settings")

temperature = st.sidebar.number_input(
    "Temperature / K",
    min_value=1.0,
    max_value=5000.0,
    value=298.15,
    step=10.0,
)

strictness_factor = st.sidebar.slider(
    "Energy strictness factor",
    min_value=1.0,
    max_value=10.0,
    value=1.0,
    step=0.5,
    help=(
        "1.0 gives a normal Boltzmann distribution. "
        "Larger values reduce the effect of energy differences. "
        "This is useful when conformer energies are considered uncertain."
    ),
)

energy_window = st.sidebar.number_input(
    "Energy window / kcal mol⁻¹",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.5,
    help="Conformers above this relative energy are assigned zero population.",
)


# =========================
# File upload
# =========================

uploaded_file = st.file_uploader(
    "Upload SDF file containing multiple conformers",
    type=["sdf"],
)

if not uploaded_file:
    st.info("Upload an SDF file to start.")
    st.stop()

mols = read_sdf(uploaded_file, sanitize=sanitize)

if len(mols) == 0:
    st.error("No valid molecules were read from the SDF file.")
    st.stop()

st.success(f"Successfully read {len(mols)} conformer(s) from the SDF file.")

first_mol = mols[0]

if show_atom_preview:
    with st.expander("Atom-index preview for the first conformer", expanded=True):
        st.write(
            "Use this table to confirm atom numbering. "
            "For GaussView/Gaussian atom numbers, use 1-based atom numbers."
        )
        atom_df = make_atom_index_preview(first_mol)
        st.dataframe(atom_df, use_container_width=True)


# =========================
# Energy property selection
# =========================

st.header("1. Select energy information")

available_props = get_available_sdf_properties(mols)

if not available_props:
    st.error(
        "No SDF properties were found. Energy values must be stored as SDF properties."
    )
    st.stop()

# Prefer common energy properties if present
preferred_energy_props = [
    "TOTAL_GIBBS_FREE_ENERGY_KCAL/MOL",
    "POTENTIAL_ENERGY_KCAL/MOL",
    "SCF_ENERGY",
    "SCF_ENERGY_HARTREE",
    "GIBBS_FREE_ENERGY",
]

default_energy_index = 0
for preferred in preferred_energy_props:
    if preferred in available_props:
        default_energy_index = available_props.index(preferred)
        break

energy_property = st.selectbox(
    "SDF property containing energy values",
    available_props,
    index=default_energy_index,
    help=(
        "Select the SDF property that contains SCF energy, potential energy, "
        "or Gibbs free energy. Do not select Boltzmann population."
    ),
)

# Guess energy unit from property name
energy_property_upper = energy_property.upper()

if "KCAL" in energy_property_upper:
    default_unit_index = 1  # kcal/mol
elif "KJ" in energy_property_upper:
    default_unit_index = 2  # kJ/mol
else:
    default_unit_index = 0  # Hartree

energy_unit = st.radio(
    "Energy unit",
    ["Hartree", "kcal/mol", "kJ/mol"],
    index=default_unit_index,
    horizontal=True,
)

energy_values = [
    extract_energy_from_property(mol, energy_property)
    for mol in mols
]

if any(x is None for x in energy_values):
    st.error(
        "Some conformers do not have valid numeric energy values in the selected property."
    )

    energy_check_df = pd.DataFrame(
        {
            "conformer_index": list(range(1, len(mols) + 1)),
            "conformer_name": [get_mol_name(mol, i) for i, mol in enumerate(mols)],
            "energy_value": energy_values,
        }
    )
    st.dataframe(energy_check_df, use_container_width=True)
    st.stop()

relative_energies_kcal = convert_energy_to_kcal_relative(
    energy_values,
    energy_unit=energy_unit,
)

populations = calculate_boltzmann_populations(
    relative_energies_kcal,
    temperature=temperature,
    strictness_factor=strictness_factor,
    energy_window=energy_window,
)

energy_df = pd.DataFrame(
    {
        "conformer_index": list(range(1, len(mols) + 1)),
        "conformer_name": [get_mol_name(mol, i) for i, mol in enumerate(mols)],
        "energy_raw": energy_values,
        "relative_energy_kcal_mol": relative_energies_kcal,
        "population": populations,
        "population_percent": [p * 100 for p in populations],
    }
)

with st.expander("Boltzmann population table", expanded=True):
    st.dataframe(energy_df, use_container_width=True)

    st.download_button(
        label="Download Boltzmann population table as CSV",
        data=convert_df_to_csv_bytes(energy_df),
        file_name="boltzmann_populations.csv",
        mime="text/csv",
    )


# =========================
# GUI input for NOE pairs
# =========================

st.header("2. Define H···H pairs")

st.write(
    "Enter the two proton atom numbers. "
    "If equivalent protons should be considered, add them using the buttons below. "
    "Only the protons relevant to the NOE correlation need to be entered."
)

pair_name = st.text_input(
    "NOE pair name",
    value=f"NOE_{len(st.session_state.noe_pairs) + 1}",
)

colA, colB = st.columns(2)

with colA:
    st.subheader("Proton A")

    proton_A = st.number_input(
        "Proton A atom number",
        min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
        step=1,
        value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
        key="proton_A_main",
    )

    if st.button("+ Equivalent proton for A"):
        st.session_state.eq_A_count += 1
        st.rerun()

    eq_A_atoms = []
    for i in range(st.session_state.eq_A_count):
        eq_A = st.number_input(
            f"Equivalent proton A {i + 1}",
            min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
            step=1,
            value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
            key=f"eq_A_{i}",
        )
        eq_A_atoms.append(eq_A)

with colB:
    st.subheader("Proton B")

    proton_B = st.number_input(
        "Proton B atom number",
        min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
        step=1,
        value=1 if numbering_mode == "0-based RDKit atom indices" else 2,
        key="proton_B_main",
    )

    if st.button("+ Equivalent proton for B"):
        st.session_state.eq_B_count += 1
        st.rerun()

    eq_B_atoms = []
    for i in range(st.session_state.eq_B_count):
        eq_B = st.number_input(
            f"Equivalent proton B {i + 1}",
            min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
            step=1,
            value=1 if numbering_mode == "0-based RDKit atom indices" else 2,
            key=f"eq_B_{i}",
        )
        eq_B_atoms.append(eq_B)

group_A_atoms = [proton_A] + eq_A_atoms
group_B_atoms = [proton_B] + eq_B_atoms

st.write("Current NOE pair to add:")

preview_df = pd.DataFrame(
    [
        {
            "pair_name": pair_name,
            "proton_A_group": ",".join(map(str, group_A_atoms)),
            "proton_B_group": ",".join(map(str, group_B_atoms)),
            "number_of_atom_pairs": len(group_A_atoms) * len(group_B_atoms),
        }
    ]
)
st.dataframe(preview_df, use_container_width=True)

col_add, col_clear_eq, col_clear_all = st.columns(3)

with col_add:
    if st.button("Add NOE pair"):
        if not pair_name.strip():
            st.warning("Please enter a NOE pair name.")
        elif len(set(group_A_atoms)) != len(group_A_atoms):
            st.warning("Duplicate atom numbers were found in Proton A group.")
        elif len(set(group_B_atoms)) != len(group_B_atoms):
            st.warning("Duplicate atom numbers were found in Proton B group.")
        elif set(group_A_atoms) & set(group_B_atoms):
            st.warning(
                "The same atom number is included in both Proton A and Proton B groups."
            )
        else:
            st.session_state.noe_pairs.append(
                {
                    "name": pair_name.strip(),
                    "group_A_atoms": group_A_atoms,
                    "group_B_atoms": group_B_atoms,
                }
            )
            reset_current_equivalent_fields()
            st.rerun()

with col_clear_eq:
    if st.button("Clear equivalent proton fields"):
        reset_current_equivalent_fields()
        st.rerun()

with col_clear_all:
    if st.button("Clear all NOE pairs"):
        st.session_state.noe_pairs = []
        reset_current_equivalent_fields()
        st.rerun()

if st.session_state.noe_pairs:
    st.subheader("Added NOE pairs")

    pair_table = pd.DataFrame(
        [
            {
                "pair_name": p["name"],
                "proton_A_group": ",".join(map(str, p["group_A_atoms"])),
                "proton_B_group": ",".join(map(str, p["group_B_atoms"])),
                "number_of_atom_pairs": len(p["group_A_atoms"]) * len(p["group_B_atoms"]),
            }
            for p in st.session_state.noe_pairs
        ]
    )

    st.dataframe(pair_table, use_container_width=True)

pair_definitions = st.session_state.noe_pairs

if not pair_definitions:
    st.info("Add at least one H···H pair.")
    st.stop()


# =========================
# NOE calculation
# =========================

st.header("3. NOE prediction")

try:
    summary_df, detailed_df, pairwise_df = calculate_noe_for_group_pairs(
        mols=mols,
        pair_definitions=pair_definitions,
        numbering_mode=numbering_mode,
        populations=populations,
        relative_energies_kcal=relative_energies_kcal,
        energy_values=energy_values,
        include_props=include_props,
    )
except Exception as e:
    st.error(f"Calculation error: {e}")
    st.stop()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Summary",
        "Conformer-level table",
        "Atom-pair details",
        "Wide tables",
        "Plots",
    ]
)

with tab1:
    st.subheader("NOE prediction summary")
    st.dataframe(summary_df, use_container_width=True)

    st.download_button(
        label="Download NOE summary as CSV",
        data=convert_df_to_csv_bytes(summary_df),
        file_name="noe_prediction_summary.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        **Interpretation**

        - `effective_NOE_distance_A_total_r6` is calculated from the total r⁻⁶ sum over all equivalent proton combinations.
        - `normalized_effective_NOE_distance_A` is calculated after dividing the r⁻⁶ sum by the number of evaluated atom pairs.
        - For comparison with ordinary single H···H distances, `normalized_effective_NOE_distance_A` is usually easier to interpret.
        - `NOE_likelihood_score_percent` is an empirical distance- and population-based score, not a true observation probability.
        """
    )

with tab2:
    st.subheader("Conformer-level group NOE table")
    st.dataframe(detailed_df, use_container_width=True)

    st.download_button(
        label="Download conformer-level NOE table as CSV",
        data=convert_df_to_csv_bytes(detailed_df),
        file_name="noe_conformer_level_table.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Atom-pair-level details")

    st.write(
        "This table shows every actual H···H atom-pair combination generated "
        "from equivalent proton inputs."
    )

    st.dataframe(pairwise_df, use_container_width=True)

    st.download_button(
        label="Download atom-pair-level table as CSV",
        data=convert_df_to_csv_bytes(pairwise_df),
        file_name="noe_atom_pair_details.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader("Wide-format tables")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Minimum H···H distance in each group / Å**")
        distance_wide = make_wide_table(
            detailed_df,
            "minimum_H_H_distance_A_in_group",
        )
        st.dataframe(distance_wide, use_container_width=True)

        st.download_button(
            label="Download distance wide table as CSV",
            data=convert_df_to_csv_bytes(distance_wide),
            file_name="noe_group_min_distances_wide.csv",
            mime="text/csv",
        )

    with col2:
        st.markdown("**Population-weighted group r⁻⁶ contributions**")
        r6_wide = make_wide_table(
            detailed_df,
            "population_weighted_group_r_minus_6_sum",
        )
        st.dataframe(r6_wide, use_container_width=True)

        st.download_button(
            label="Download r-6 contribution wide table as CSV",
            data=convert_df_to_csv_bytes(r6_wide),
            file_name="noe_group_r6_contributions_wide.csv",
            mime="text/csv",
        )

with tab5:
    st.subheader("Plots")

    if not detailed_df.empty:
        st.markdown("**Minimum H···H distance in each group by conformer**")

        distance_chart = (
            alt.Chart(detailed_df)
            .mark_circle(size=70)
            .encode(
                x=alt.X("conformer_index:Q", title="Conformer index"),
                y=alt.Y(
                    "minimum_H_H_distance_A_in_group:Q",
                    title="Minimum H···H distance in group / Å",
                ),
                color=alt.Color("pair_name:N", title="NOE pair"),
                tooltip=[
                    "pair_name",
                    "group_A_atoms",
                    "group_B_atoms",
                    "conformer_index",
                    "conformer_name",
                    alt.Tooltip("minimum_H_H_distance_A_in_group:Q", format=".3f"),
                    alt.Tooltip("relative_energy_kcal_mol:Q", format=".3f"),
                    alt.Tooltip("population_percent:Q", format=".2f"),
                ],
            )
            .interactive()
        )

        st.altair_chart(distance_chart, use_container_width=True)

        st.markdown("**Population-weighted group r⁻⁶ contribution by conformer**")

        r6_chart = (
            alt.Chart(detailed_df)
            .mark_bar()
            .encode(
                x=alt.X("conformer_index:O", title="Conformer index"),
                y=alt.Y(
                    "population_weighted_group_r_minus_6_sum:Q",
                    title="Population-weighted group r⁻⁶ sum",
                ),
                color=alt.Color("pair_name:N", title="NOE pair"),
                tooltip=[
                    "pair_name",
                    "group_A_atoms",
                    "group_B_atoms",
                    "conformer_index",
                    "conformer_name",
                    alt.Tooltip(
                        "population_weighted_group_r_minus_6_sum:Q",
                        format=".6g",
                    ),
                    alt.Tooltip("population_percent:Q", format=".2f"),
                ],
            )
            .interactive()
        )

        st.altair_chart(r6_chart, use_container_width=True)

    if not pairwise_df.empty:
        st.markdown("**Atom-pair-level H···H distances**")

        pairwise_chart = (
            alt.Chart(pairwise_df)
            .mark_circle(size=50)
            .encode(
                x=alt.X("conformer_index:Q", title="Conformer index"),
                y=alt.Y("H_H_distance_A:Q", title="H···H distance / Å"),
                color=alt.Color("atom_pair_input:N", title="Atom pair"),
                tooltip=[
                    "pair_name",
                    "atom_pair_input",
                    "conformer_index",
                    "conformer_name",
                    alt.Tooltip("H_H_distance_A:Q", format=".3f"),
                    alt.Tooltip("population_percent:Q", format=".2f"),
                ],
            )
            .interactive()
        )

        st.altair_chart(pairwise_chart, use_container_width=True)

st.divider()

st.caption(
    "Notes: This app predicts NOE likelihood from conformer populations and H···H distances. "
    "Equivalent proton inputs are treated by summing r⁻⁶ contributions over all specified atom-pair combinations. "
    "The app does not explicitly account for spin diffusion, mixing time, molecular correlation time, signal overlap, "
    "exchangeable protons, or relaxation mechanisms. Use the output as a conformational and geometrical guide, "
    "not as definitive proof."
)
