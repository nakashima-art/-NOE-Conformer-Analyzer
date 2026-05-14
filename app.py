import re
from io import StringIO
import math

import altair as alt
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


APP_VERSION = "ver. 1.0"

R_KCAL = 0.00198720425864083  # kcal mol^-1 K^-1


# =========================
# Utility functions
# =========================

def parse_pair_definitions(text):
    """
    Parse H-H pair definitions.

    Accepted formats:
        NOE_1: 12-35
        H3_H8: 3, 8
        10 25

    Atom numbers are interpreted according to the selected numbering mode.
    """
    definitions = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if ":" in line:
            name_part, atom_part = line.split(":", 1)
            name = name_part.strip()
        else:
            name = f"NOE_pair_{len(definitions) + 1}"
            atom_part = line

        nums = re.findall(r"\d+", atom_part)

        if len(nums) != 2:
            raise ValueError(
                f"Line {line_no}: '{line}' must contain exactly two atom numbers."
            )

        atoms_input = [int(nums[0]), int(nums[1])]

        definitions.append(
            {
                "name": name,
                "atoms_input": atoms_input,
                "line_no": line_no,
            }
        )

    return definitions


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
    sdf_text = uploaded_file.getvalue().decode("utf-8", errors="replace")

    supplier = Chem.ForwardSDMolSupplier(
        StringIO(sdf_text),
        sanitize=sanitize,
        removeHs=False,
    )

    mols = [mol for mol in supplier if mol is not None]

    return mols


def get_available_sdf_properties(mols):
    prop_names = set()

    for mol in mols:
        for prop_name in mol.GetPropNames():
            prop_names.add(prop_name)

    return sorted(prop_names)


def extract_energy_from_property(mol, property_name):
    """
    Extract numeric energy from an SDF property.
    The property value may contain text, but the first numeric value is used.
    """
    if not mol.HasProp(property_name):
        return None

    value = mol.GetProp(property_name).strip()

    match = re.search(
        r"[-+]?\d*\.?\d+(?:[EeDd][-+]?\d+)?",
        value
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
        1.0 = normal Boltzmann
        >1.0 = softened energy penalty
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

    populations = [w / total for w in weights]

    return populations


def distance_to_score(distance):
    """
    Empirical NOE likelihood score for each conformer.
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


def calculate_noe_for_pairs(
    mols,
    pair_definitions,
    numbering_mode,
    populations,
    relative_energies_kcal,
    energy_values,
    include_props,
):
    """
    Calculate H-H distances, r^-6 terms, and weighted contributions.
    """
    detailed_rows = []

    for conf_idx, mol in enumerate(mols):
        if mol is None or mol.GetNumConformers() == 0:
            continue

        conf = mol.GetConformer()
        mol_name = get_mol_name(mol, conf_idx)
        props = mol_properties_to_dict(mol) if include_props else {}

        for pair in pair_definitions:
            atom_indices = convert_to_zero_based(
                pair["atoms_input"],
                numbering_mode,
            )

            if any(idx < 0 or idx >= mol.GetNumAtoms() for idx in atom_indices):
                raise ValueError(
                    f"Invalid atom number in {pair['name']}: "
                    f"{pair['atoms_input']} for molecule {mol_name}. "
                    f"This molecule has {mol.GetNumAtoms()} atoms."
                )

            distance = rdMolTransforms.GetBondLength(
                conf,
                atom_indices[0],
                atom_indices[1],
            )

            if distance <= 0:
                r_minus_6 = None
                weighted_r_minus_6 = None
            else:
                r_minus_6 = distance ** -6
                weighted_r_minus_6 = populations[conf_idx] * r_minus_6

            score = distance_to_score(distance)
            weighted_score = populations[conf_idx] * score

            row = {
                "pair_name": pair["name"],
                "atoms_input": "-".join(map(str, pair["atoms_input"])),
                "atoms_0_based": "-".join(map(str, atom_indices)),
                "conformer_index": conf_idx + 1,
                "conformer_name": mol_name,
                "energy_raw": energy_values[conf_idx],
                "relative_energy_kcal_mol": relative_energies_kcal[conf_idx],
                "population": populations[conf_idx],
                "population_percent": populations[conf_idx] * 100,
                "H_H_distance_A": distance,
                "r_minus_6": r_minus_6,
                "population_weighted_r_minus_6": weighted_r_minus_6,
                "conformer_distance_score": score,
                "population_weighted_score": weighted_score,
            }

            row.update(props)
            detailed_rows.append(row)

    detailed_df = pd.DataFrame(detailed_rows)

    summary_rows = []

    for pair_name, group in detailed_df.groupby("pair_name"):
        weighted_r6_sum = group["population_weighted_r_minus_6"].sum()

        if weighted_r6_sum > 0:
            r_eff = weighted_r6_sum ** (-1 / 6)
        else:
            r_eff = None

        likelihood_score = group["population_weighted_score"].sum() * 100

        if r_eff is not None:
            prediction = classify_effective_distance(r_eff)
        else:
            prediction = "not calculated"

        min_distance = group["H_H_distance_A"].min()
        max_population = group["population_percent"].max()

        closest_row = group.loc[group["H_H_distance_A"].idxmin()]
        largest_contribution_row = group.loc[
            group["population_weighted_r_minus_6"].idxmax()
        ]

        summary_rows.append(
            {
                "pair_name": pair_name,
                "atoms_input": closest_row["atoms_input"],
                "effective_NOE_distance_A": r_eff,
                "NOE_likelihood_score_percent": likelihood_score,
                "prediction": prediction,
                "minimum_H_H_distance_A": min_distance,
                "highest_population_percent": max_population,
                "closest_conformer": closest_row["conformer_name"],
                "closest_conformer_distance_A": closest_row["H_H_distance_A"],
                "largest_NOE_contribution_conformer": largest_contribution_row["conformer_name"],
                "largest_NOE_contribution_percent_of_total_r6": (
                    largest_contribution_row["population_weighted_r_minus_6"]
                    / weighted_r6_sum
                    * 100
                    if weighted_r6_sum > 0
                    else None
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, detailed_df


def make_wide_table(detailed_df, value_column):
    if detailed_df.empty:
        return pd.DataFrame()

    wide = detailed_df.pivot_table(
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
    "and evaluates population-weighted r⁻⁶ effective distances."
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
    "Include SDF properties in detailed output",
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
        "Larger values reduce the effect of energy differences, which is useful "
        "when conformer energies are considered uncertain."
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

energy_property = st.selectbox(
    "SDF property containing energy values",
    available_props,
    help=(
        "Select the SDF property that contains SCF energy or Gibbs free energy. "
        "The first numeric value in the property is used."
    ),
)

energy_unit = st.radio(
    "Energy unit",
    ["Hartree", "kcal/mol", "kJ/mol"],
    index=0,
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
# H-H pair definitions
# =========================

st.header("2. Define H···H pairs")

pair_text = st.text_area(
    "H···H pair definitions",
    value="NOE_1: 12-35\n# NOE_2: 18, 42",
    height=140,
    help=(
        "Define one proton pair per line using two atom numbers. "
        "Examples: NOE_1: 12-35 or H3_H8: 3, 8."
    ),
)

try:
    pair_definitions = parse_pair_definitions(pair_text)
except Exception as e:
    st.error(f"Error in H···H pair definitions: {e}")
    st.stop()

if not pair_definitions:
    st.info("Define at least one H···H pair.")
    st.stop()

with st.expander("Parsed H···H pair definitions", expanded=False):
    st.dataframe(pd.DataFrame(pair_definitions), use_container_width=True)

# =========================
# NOE calculation
# =========================

st.header("3. NOE prediction")

try:
    summary_df, detailed_df = calculate_noe_for_pairs(
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

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Summary",
        "Detailed conformer table",
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

        - `effective_NOE_distance_A` is calculated as  
          \( r_{eff} = (\\sum p_i r_i^{-6})^{-1/6} \).
        - `NOE_likelihood_score_percent` is an empirical distance- and population-based score.
        - The score is **not a true observation probability**, but a practical indicator.
        """
    )

with tab2:
    st.subheader("Detailed conformer-level NOE table")
    st.dataframe(detailed_df, use_container_width=True)

    st.download_button(
        label="Download detailed NOE table as CSV",
        data=convert_df_to_csv_bytes(detailed_df),
        file_name="noe_detailed_conformer_table.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Wide-format tables")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**H···H distances / Å**")
        distance_wide = make_wide_table(detailed_df, "H_H_distance_A")
        st.dataframe(distance_wide, use_container_width=True)

        st.download_button(
            label="Download distance wide table as CSV",
            data=convert_df_to_csv_bytes(distance_wide),
            file_name="noe_distances_wide.csv",
            mime="text/csv",
        )

    with col2:
        st.markdown("**Population-weighted r⁻⁶ contributions**")
        r6_wide = make_wide_table(detailed_df, "population_weighted_r_minus_6")
        st.dataframe(r6_wide, use_container_width=True)

        st.download_button(
            label="Download r-6 contribution wide table as CSV",
            data=convert_df_to_csv_bytes(r6_wide),
            file_name="noe_r6_contributions_wide.csv",
            mime="text/csv",
        )

with tab4:
    st.subheader("Plots")

    st.markdown("**H···H distance by conformer**")

    distance_chart = (
        alt.Chart(detailed_df)
        .mark_circle(size=70)
        .encode(
            x=alt.X("conformer_index:Q", title="Conformer index"),
            y=alt.Y("H_H_distance_A:Q", title="H···H distance / Å"),
            color=alt.Color("pair_name:N", title="H···H pair"),
            tooltip=[
                "pair_name",
                "conformer_index",
                "conformer_name",
                alt.Tooltip("H_H_distance_A:Q", format=".3f"),
                alt.Tooltip("relative_energy_kcal_mol:Q", format=".3f"),
                alt.Tooltip("population_percent:Q", format=".2f"),
            ],
        )
        .interactive()
    )

    st.altair_chart(distance_chart, use_container_width=True)

    st.markdown("**Population-weighted r⁻⁶ contribution by conformer**")

    r6_chart = (
        alt.Chart(detailed_df)
        .mark_bar()
        .encode(
            x=alt.X("conformer_index:O", title="Conformer index"),
            y=alt.Y(
                "population_weighted_r_minus_6:Q",
                title="Population-weighted r⁻⁶",
            ),
            color=alt.Color("pair_name:N", title="H···H pair"),
            tooltip=[
                "pair_name",
                "conformer_index",
                "conformer_name",
                alt.Tooltip("H_H_distance_A:Q", format=".3f"),
                alt.Tooltip("population_weighted_r_minus_6:Q", format=".6g"),
                alt.Tooltip("population_percent:Q", format=".2f"),
            ],
        )
        .interactive()
    )

    st.altair_chart(r6_chart, use_container_width=True)

st.divider()

st.caption(
    "Notes: This app predicts NOE likelihood from conformer populations and "
    "H···H distances. It does not explicitly account for spin diffusion, mixing time, "
    "molecular correlation time, signal overlap, exchangeable protons, or relaxation mechanisms. "
    "Use the output as a conformational and geometrical guide, not as a definitive proof."
)
