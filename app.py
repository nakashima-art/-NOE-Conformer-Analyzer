import json
import math
import re
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import rdMolTransforms


APP_VERSION = "ver. 1.5"
R_KCAL = 0.00198720425864083  # kcal mol^-1 K^-1


# =========================
# Language settings
# =========================

LANG = {
    "English": {
        "app_title": "NOE Conformer Analyzer",
        "app_description": (
            "This app estimates whether NOE correlations are likely to be observed "
            "from conformer geometries in an SDF file. It calculates H···H distances "
            "for each conformer, derives Boltzmann populations from SDF energy properties, "
            "and evaluates population-weighted r⁻⁶ effective distances. "
            "Equivalent protons can be added only when needed."
        ),
        "language": "Language / 言語",
        "input_settings": "Input settings",
        "atom_numbering": "Atom numbering in your input",
        "sanitize": "Sanitize SDF while reading",
        "include_props": "Include SDF properties in detailed outputs",
        "show_atom_preview": "Show atom-index table for the first conformer",
        "show_3d_viewer": "Show 3D atom-numbering viewer",
        "label_mode": "3D viewer atom labels",
        "viewer_height": "3D viewer height",
        "boltzmann_settings": "Boltzmann settings",
        "temperature": "Temperature / K",
        "strictness_factor": "Energy strictness factor",
        "energy_window": "Energy window / kcal mol⁻¹",
        "advanced_analysis": "Advanced analysis",
        "run_sensitivity": "Run strictness sensitivity analysis",
        "upload_sdf": "Upload SDF file containing multiple conformers",
        "upload_prompt": "Upload an SDF file to start.",
        "read_success": "Successfully read {n} conformer(s) from the SDF file.",
        "no_valid_mols": "No valid molecules were read from the SDF file.",
        "atom_helper": "Atom numbering helper",
        "viewer_title": "3D viewer of the first conformer",
        "viewer_help": (
            "Use this 3D viewer to confirm atom numbers without opening external software. "
            "The displayed structure is the first conformer in the uploaded SDF file."
        ),
        "viewer_tip": (
            "Tip: Use 'H atoms only' for NOE analysis. "
            "If labels overlap, rotate or zoom the model in the viewer."
        ),
        "atom_table_title": "Atom-index table for the first conformer",
        "atom_table_help": "This table can be used together with the 3D viewer to confirm atom numbering.",
        "energy_info": "1. Select energy information",
        "no_energy_props": (
            "No appropriate energy-like SDF properties were found. "
            "Energy values must be stored as SDF properties."
        ),
        "energy_property": "SDF property containing energy values",
        "population_selected_error": (
            "The selected property appears to be a population, not an energy. "
            "Please select an energy property such as POTENTIAL_ENERGY_KCAL/MOL "
            "or TOTAL_GIBBS_FREE_ENERGY_KCAL/MOL."
        ),
        "energy_unit": "Energy unit",
        "invalid_energy_values": "Some conformers do not have valid numeric energy values in the selected property.",
        "boltzmann_table": "Boltzmann population table",
        "download_boltzmann": "Download Boltzmann population table as CSV",
        "define_pairs": "2. Define H···H pairs",
        "define_pairs_help": (
            "Enter the two proton atom numbers. "
            "If equivalent protons should be considered, add them using the buttons below."
        ),
        "pair_name": "NOE pair name",
        "proton_a": "Proton A",
        "proton_b": "Proton B",
        "proton_a_number": "Proton A atom number",
        "proton_b_number": "Proton B atom number",
        "eq_a": "+ Equivalent proton for A",
        "eq_b": "+ Equivalent proton for B",
        "eq_a_label": "Equivalent proton A {i}",
        "eq_b_label": "Equivalent proton B {i}",
        "current_pair": "Current NOE pair to add:",
        "add_pair": "Add NOE pair",
        "clear_eq": "Clear equivalent proton fields",
        "added_pairs": "Added NOE pairs",
        "delete_selected_title": "Delete selected NOE pair",
        "select_pair_delete": "Select a NOE pair to delete",
        "delete_selected": "Delete selected NOE pair",
        "clear_all": "Clear all NOE pairs",
        "enter_pair_name_warning": "Please enter a NOE pair name.",
        "duplicate_a_warning": "Duplicate atom numbers were found in Proton A group.",
        "duplicate_b_warning": "Duplicate atom numbers were found in Proton B group.",
        "overlap_warning": "The same atom number is included in both Proton A and Proton B groups.",
        "deleted_pair": "Deleted NOE pair: {name}",
        "add_one_pair": "Add at least one H···H pair.",
        "prediction": "3. NOE prediction",
        "calc_error": "Calculation error: {e}",
        "sensitivity_failed": "Strictness sensitivity analysis failed: {e}",
        "summary": "Summary",
        "conformer_table": "Conformer-level table",
        "atom_pair_details": "Atom-pair details",
        "wide_tables": "Wide tables",
        "plots": "Plots",
        "strictness_sensitivity": "Strictness sensitivity",
        "summary_title": "NOE prediction summary",
        "download_summary": "Download NOE summary as CSV",
        "intuitive_interpretation": "Intuitive NOE likelihood interpretation",
        "noe_score": "NOE likelihood score",
        "norm_dist": "Normalized effective distance",
        "min_dist": "Minimum H···H distance",
        "contact_population": "Contact population",
        "robustness_index": "Robustness index",
        "largest_r6": "Largest single r⁻⁶ contribution",
        "score_range": "Score range",
        "distance_range": "Distance range",
        "interpretation_notes": "Interpretation notes",
        "conformer_table_title": "Conformer-level group NOE table",
        "download_conformer": "Download conformer-level NOE table as CSV",
        "atom_pair_details_title": "Atom-pair-level details",
        "atom_pair_details_help": (
            "This table shows every actual H···H atom-pair combination generated "
            "from equivalent proton inputs."
        ),
        "download_pairwise": "Download atom-pair-level table as CSV",
        "wide_tables_title": "Wide-format tables",
        "min_distance_wide": "Minimum H···H distance in each group / Å",
        "r6_wide": "Population-weighted group r⁻⁶ contributions",
        "download_distance_wide": "Download distance wide table as CSV",
        "download_r6_wide": "Download r-6 contribution wide table as CSV",
        "plots_title": "Plots",
        "plot_min_distance": "Minimum H···H distance in each group by conformer",
        "plot_r6": "Population-weighted group r⁻⁶ contribution by conformer",
        "plot_pairwise": "Atom-pair-level H···H distances",
        "sensitivity_title": "Strictness sensitivity analysis",
        "sensitivity_off": "Strictness sensitivity analysis is turned off in the sidebar.",
        "sensitivity_none": "No strictness sensitivity results are available.",
        "sensitivity_help": (
            "This analysis recalculates NOE metrics using energy strictness factors "
            "1.0, 2.0, and 5.0. Larger differences indicate stronger dependence on "
            "the conformer population model."
        ),
        "sensitivity_summary": "Sensitivity summary",
        "sensitivity_details": "Sensitivity details",
        "download_sensitivity_summary": "Download strictness sensitivity summary as CSV",
        "download_sensitivity_details": "Download strictness sensitivity details as CSV",
        "caption": (
            "Notes: This app predicts NOE likelihood from conformer populations and H···H distances. "
            "Equivalent proton inputs are treated by summing r⁻⁶ contributions over all specified atom-pair combinations. "
            "Contact population, robustness index, and strictness sensitivity are intended to help evaluate reliability. "
            "The 3D viewer is used only for atom-number confirmation; NOE calculations use the original 3D coordinates "
            "stored in the uploaded SDF file. "
            "The app does not explicitly account for spin diffusion, mixing time, molecular correlation time, signal overlap, "
            "exchangeable protons, or relaxation mechanisms. Use the output as a conformational and geometrical guide, "
            "not as definitive proof."
        ),
    },
    "日本語": {
        "app_title": "NOE配座解析アプリ",
        "app_description": (
            "このアプリは、複数配座を含むSDFファイルから各配座のH···H距離を計算し、"
            "SDF property中のエネルギー値からBoltzmann存在比を算出したうえで、"
            "r⁻⁶重み付き実効距離に基づいてNOE相関が観測される可能性を評価します。"
            "必要な場合のみ等価プロトンを追加できます。"
        ),
        "language": "Language / 言語",
        "input_settings": "入力設定",
        "atom_numbering": "入力する原子番号の形式",
        "sanitize": "SDF読み込み時にsanitizeする",
        "include_props": "詳細出力にSDF propertyを含める",
        "show_atom_preview": "最初の配座の原子番号表を表示",
        "show_3d_viewer": "3D原子番号確認ビューアを表示",
        "label_mode": "3Dビューアの原子ラベル",
        "viewer_height": "3Dビューアの高さ",
        "boltzmann_settings": "Boltzmann分布設定",
        "temperature": "温度 / K",
        "strictness_factor": "エネルギー厳格度係数",
        "energy_window": "エネルギー窓 / kcal mol⁻¹",
        "advanced_analysis": "詳細解析",
        "run_sensitivity": "厳格度感度解析を実行",
        "upload_sdf": "複数配座を含むSDFファイルをアップロード",
        "upload_prompt": "SDFファイルをアップロードしてください。",
        "read_success": "SDFファイルから {n} 個の配座を読み込みました。",
        "no_valid_mols": "有効な分子をSDFファイルから読み込めませんでした。",
        "atom_helper": "原子番号確認",
        "viewer_title": "最初の配座の3D表示",
        "viewer_help": (
            "外部ソフトを開かずに原子番号を確認するための3Dビューアです。"
            "アップロードされたSDFファイル中の最初の配座を表示しています。"
        ),
        "viewer_tip": (
            "NOE解析では通常 'H atoms only' が見やすいです。"
            "ラベルが重なる場合は、ビューア内で回転・拡大縮小してください。"
        ),
        "atom_table_title": "最初の配座の原子番号表",
        "atom_table_help": "3Dビューアと併用して原子番号を確認できます。",
        "energy_info": "1. エネルギー情報の選択",
        "no_energy_props": (
            "適切なエネルギー様SDF propertyが見つかりませんでした。"
            "エネルギー値はSDF propertyとして保存されている必要があります。"
        ),
        "energy_property": "エネルギー値を含むSDF property",
        "population_selected_error": (
            "選択されたpropertyは存在比であり、エネルギーではない可能性があります。"
            "POTENTIAL_ENERGY_KCAL/MOL や TOTAL_GIBBS_FREE_ENERGY_KCAL/MOL などの"
            "エネルギーpropertyを選択してください。"
        ),
        "energy_unit": "エネルギー単位",
        "invalid_energy_values": "選択されたpropertyに有効な数値エネルギーを持たない配座があります。",
        "boltzmann_table": "Boltzmann存在比表",
        "download_boltzmann": "Boltzmann存在比表をCSVでダウンロード",
        "define_pairs": "2. H···Hペアの指定",
        "define_pairs_help": (
            "NOEを評価したい2つのプロトン原子番号を入力してください。"
            "等価プロトンを考慮する場合は、下のボタンで追加できます。"
        ),
        "pair_name": "NOEペア名",
        "proton_a": "プロトンA",
        "proton_b": "プロトンB",
        "proton_a_number": "プロトンAの原子番号",
        "proton_b_number": "プロトンBの原子番号",
        "eq_a": "+ A側の等価プロトンを追加",
        "eq_b": "+ B側の等価プロトンを追加",
        "eq_a_label": "A側の等価プロトン {i}",
        "eq_b_label": "B側の等価プロトン {i}",
        "current_pair": "追加予定のNOEペア:",
        "add_pair": "NOEペアを追加",
        "clear_eq": "等価プロトン入力欄をクリア",
        "added_pairs": "追加済みNOEペア",
        "delete_selected_title": "選択したNOEペアを削除",
        "select_pair_delete": "削除するNOEペアを選択",
        "delete_selected": "選択したNOEペアを削除",
        "clear_all": "すべてのNOEペアを削除",
        "enter_pair_name_warning": "NOEペア名を入力してください。",
        "duplicate_a_warning": "プロトンA側に重複した原子番号があります。",
        "duplicate_b_warning": "プロトンB側に重複した原子番号があります。",
        "overlap_warning": "同じ原子番号がプロトンA側とB側の両方に含まれています。",
        "deleted_pair": "削除したNOEペア: {name}",
        "add_one_pair": "少なくとも1つのH···Hペアを追加してください。",
        "prediction": "3. NOE予測",
        "calc_error": "計算エラー: {e}",
        "sensitivity_failed": "厳格度感度解析に失敗しました: {e}",
        "summary": "概要",
        "conformer_table": "配座ごとの詳細",
        "atom_pair_details": "原子ペアごとの詳細",
        "wide_tables": "ワイド形式表",
        "plots": "プロット",
        "strictness_sensitivity": "厳格度感度解析",
        "summary_title": "NOE予測概要",
        "download_summary": "NOE概要をCSVでダウンロード",
        "intuitive_interpretation": "NOE観測可能性の直感的判定",
        "noe_score": "NOE観測可能性スコア",
        "norm_dist": "規格化実効距離",
        "min_dist": "最短H···H距離",
        "contact_population": "近距離配座存在比",
        "robustness_index": "頑健性指標",
        "largest_r6": "最大単一r⁻⁶寄与",
        "score_range": "スコア変動幅",
        "distance_range": "距離変動幅",
        "interpretation_notes": "解釈上の注意",
        "conformer_table_title": "配座ごとのグループNOE詳細表",
        "download_conformer": "配座ごとのNOE詳細表をCSVでダウンロード",
        "atom_pair_details_title": "原子ペアごとの詳細",
        "atom_pair_details_help": (
            "等価プロトン入力から生成された、実際のすべてのH···H原子ペアを表示します。"
        ),
        "download_pairwise": "原子ペアごとの詳細表をCSVでダウンロード",
        "wide_tables_title": "ワイド形式表",
        "min_distance_wide": "各グループ内の最短H···H距離 / Å",
        "r6_wide": "存在比重み付きグループr⁻⁶寄与",
        "download_distance_wide": "距離ワイド表をCSVでダウンロード",
        "download_r6_wide": "r⁻⁶寄与ワイド表をCSVでダウンロード",
        "plots_title": "プロット",
        "plot_min_distance": "配座ごとのグループ内最短H···H距離",
        "plot_r6": "配座ごとの存在比重み付きグループr⁻⁶寄与",
        "plot_pairwise": "原子ペアごとのH···H距離",
        "sensitivity_title": "厳格度感度解析",
        "sensitivity_off": "サイドバーで厳格度感度解析がオフになっています。",
        "sensitivity_none": "厳格度感度解析の結果がありません。",
        "sensitivity_help": (
            "エネルギー厳格度係数 1.0、2.0、5.0 でNOE指標を再計算します。"
            "差が大きいほど、配座存在比モデルへの依存性が高いことを示します。"
        ),
        "sensitivity_summary": "感度解析概要",
        "sensitivity_details": "感度解析詳細",
        "download_sensitivity_summary": "厳格度感度解析概要をCSVでダウンロード",
        "download_sensitivity_details": "厳格度感度解析詳細をCSVでダウンロード",
        "caption": (
            "注: このアプリは、配座存在比とH···H距離からNOE観測可能性を予測するための補助ツールです。"
            "等価プロトンを指定した場合は、指定されたすべての原子ペアのr⁻⁶寄与を合算します。"
            "近距離配座存在比、頑健性指標、厳格度感度解析は、予測の信頼性を評価するための補助指標です。"
            "3Dビューアは原子番号確認用であり、NOE計算にはアップロードされたSDF中の元の3D座標を使用します。"
            "スピン拡散、mixing time、分子の相関時間、シグナル重なり、交換性プロトン、緩和機構は明示的には考慮していません。"
            "本結果は構造・配座に基づく幾何学的な判断材料として扱い、決定的証拠としては扱わないでください。"
        ),
    },
}


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
    prop_names = set()

    for mol in mols:
        for prop_name in mol.GetPropNames():
            prop_names.add(prop_name)

    prop_names = sorted(prop_names)

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

    if not energy_props:
        energy_props = [
            prop for prop in prop_names
            if not any(keyword in prop.upper() for keyword in excluded_keywords)
        ]

    return energy_props


def extract_energy_from_property(mol, property_name):
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
# 3D viewer functions
# =========================

def make_atom_labels_for_3dmol(mol, numbering_mode, label_mode):
    if mol.GetNumConformers() == 0:
        return []

    conf = mol.GetConformer()
    labels = []

    for atom in mol.GetAtoms():
        idx0 = atom.GetIdx()
        element = atom.GetSymbol()

        if label_mode == "No labels":
            continue

        if label_mode == "H atoms only" and element != "H":
            continue

        atom_number = idx0 + 1 if numbering_mode == "1-based atom numbers" else idx0
        pos = conf.GetAtomPosition(idx0)

        labels.append(
            {
                "x": float(pos.x),
                "y": float(pos.y),
                "z": float(pos.z),
                "label": f"{element}{atom_number}",
                "element": element,
                "atom_number": atom_number,
            }
        )

    return labels


def render_3dmol_viewer(mol, numbering_mode, label_mode, viewer_height=520):
    molblock = Chem.MolToMolBlock(mol)
    labels = make_atom_labels_for_3dmol(
        mol=mol,
        numbering_mode=numbering_mode,
        label_mode=label_mode,
    )

    molblock_js = json.dumps(molblock)
    labels_js = json.dumps(labels)

    html = f"""
    <div id="viewer3dmol" style="width: 100%; height: {viewer_height}px; position: relative;"></div>

    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
    const molblock = {molblock_js};
    const labels = {labels_js};

    function makeViewer() {{
        let element = document.getElementById("viewer3dmol");
        let viewer = $3Dmol.createViewer(element, {{
            backgroundColor: "white"
        }});

        viewer.addModel(molblock, "sdf");

        viewer.setStyle({{}}, {{
            stick: {{
                radius: 0.15
            }},
            sphere: {{
                scale: 0.22
            }}
        }});

        labels.forEach(function(atom) {{
            viewer.addLabel(atom.label, {{
                position: {{
                    x: atom.x,
                    y: atom.y,
                    z: atom.z
                }},
                fontSize: 12,
                fontColor: "black",
                backgroundColor: "white",
                backgroundOpacity: 0.75,
                borderThickness: 0.5,
                borderColor: "black",
                inFront: true,
                showBackground: true
            }});
        }});

        viewer.zoomTo();
        viewer.render();
    }}

    if (typeof $3Dmol !== "undefined") {{
        makeViewer();
    }} else {{
        document.getElementById("viewer3dmol").innerHTML =
            "<p style='color:red;'>3Dmol.js could not be loaded.</p>";
    }}
    </script>
    """

    components.html(html, height=viewer_height + 20, scrolling=False)


# =========================
# NOE scoring functions
# =========================

def distance_to_score(distance):
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
    if r_eff is None or pd.isna(r_eff):
        return "not calculated"
    if r_eff <= 2.5:
        return "strong NOE likely"
    if r_eff <= 3.0:
        return "moderate NOE likely"
    if r_eff <= 3.5:
        return "weak/possible NOE"
    if r_eff <= 4.0:
        return "borderline"
    return "unlikely"


def interpret_noe_likelihood(score_percent, normalized_distance, language="English"):
    if language == "日本語":
        if normalized_distance is not None and pd.notna(normalized_distance):
            if normalized_distance > 4.0:
                return {
                    "level": "NOEの観測は困難",
                    "style": "error",
                    "comment": (
                        "規格化実効H···H距離が4.0 Åを超えています。"
                        "通常の条件では明瞭なNOE相関は観測されにくいと考えられます。"
                    ),
                }

        if score_percent >= 80:
            return {
                "level": "強いNOEが期待される",
                "style": "success",
                "comment": (
                    "このプロトンペアではNOE相関が観測される可能性が高く、"
                    "比較的明瞭なNOEが期待されます。"
                ),
            }

        if score_percent >= 60:
            return {
                "level": "中程度のNOEが期待される",
                "style": "success",
                "comment": (
                    "このプロトンペアでは中程度のNOE相関が観測される可能性があります。"
                    "実験条件によっては十分に観測可能と考えられます。"
                ),
            }

        if score_percent >= 30:
            return {
                "level": "弱いNOEの可能性",
                "style": "info",
                "comment": (
                    "このプロトンペアでは弱いNOE相関が観測される可能性がありますが、"
                    "強いNOEは期待しにくいです。実効距離、近距離配座存在比、"
                    "頑健性指標と併せて解釈してください。"
                ),
            }

        if score_percent >= 10:
            return {
                "level": "境界的",
                "style": "warning",
                "comment": (
                    "このプロトンペアは境界的です。NOE相関は弱い、または"
                    "実験感度によっては観測が難しい可能性があります。"
                ),
            }

        return {
            "level": "NOEの観測は困難",
            "style": "error",
            "comment": (
                "NOE観測可能性スコアが低いため、通常の条件ではNOE相関は"
                "観測されにくいと考えられます。"
            ),
        }

    if normalized_distance is not None and pd.notna(normalized_distance):
        if normalized_distance > 4.0:
            return {
                "level": "NOE unlikely",
                "style": "error",
                "comment": (
                    "The normalized effective H···H distance is longer than 4.0 Å. "
                    "A clear NOE correlation is unlikely under typical conditions."
                ),
            }

    if score_percent >= 80:
        return {
            "level": "Strong NOE likely",
            "style": "success",
            "comment": (
                "This proton pair is predicted to have a high likelihood of showing "
                "a NOE correlation. A relatively clear NOE may be expected."
            ),
        }

    if score_percent >= 60:
        return {
            "level": "Moderate NOE likely",
            "style": "success",
            "comment": (
                "This proton pair is predicted to have a moderate likelihood of showing "
                "a NOE correlation. The NOE may be observable depending on experimental conditions."
            ),
        }

    if score_percent >= 30:
        return {
            "level": "Weak or possible NOE",
            "style": "info",
            "comment": (
                "This proton pair may show a weak NOE correlation, but a strong NOE is not expected. "
                "The result should be interpreted together with the effective distance, contact population, "
                "and robustness index."
            ),
        }

    if score_percent >= 10:
        return {
            "level": "Borderline",
            "style": "warning",
            "comment": (
                "This proton pair is borderline. A NOE correlation may be weak or difficult to observe, "
                "especially if the experimental sensitivity is limited."
            ),
        }

    return {
        "level": "NOE unlikely",
        "style": "error",
        "comment": (
            "This proton pair has a low NOE likelihood score. A NOE correlation is unlikely "
            "under typical conditions."
        ),
    }


def show_interpretation_box(interpretation):
    text = f"**{interpretation['level']}**\n\n{interpretation['comment']}"

    if interpretation["style"] == "success":
        st.success(text)
    elif interpretation["style"] == "info":
        st.info(text)
    elif interpretation["style"] == "warning":
        st.warning(text)
    else:
        st.error(text)


def interpret_contact_population(p30, p35, p40, language="English"):
    if language == "日本語":
        if p30 >= 50:
            return "存在比の大きい配座の多くでH···H距離が3.0 Å以内にあり、NOE予測は幾何学的に比較的強く支持されます。"
        if p35 >= 50:
            return "配座集合のかなりの割合が弱〜中程度のNOE距離範囲にあります。"
        if p40 >= 50:
            return "多くの配座は境界的なNOE距離範囲にありますが、明確な近接配座は限定的です。"
        return "近接したH···H接触を示す配座の存在比は小さく、NOE予測は少数配座に依存している可能性があります。"

    if p30 >= 50:
        return "Many populated conformers have H···H distances within 3.0 Å. The NOE prediction is geometrically well supported."
    if p35 >= 50:
        return "A substantial fraction of the conformer ensemble falls within the weak-to-moderate NOE distance range."
    if p40 >= 50:
        return "Many conformers are within the borderline NOE range, but close-contact conformers are limited."
    return "Only a small fraction of the conformer ensemble shows close H···H contact. The NOE prediction may depend on minor conformers."


def interpret_robustness(largest_contribution_percent, language="English"):
    if largest_contribution_percent is None or pd.isna(largest_contribution_percent):
        return "頑健性を評価できませんでした。" if language == "日本語" else "Robustness could not be evaluated."

    if language == "日本語":
        if largest_contribution_percent < 20:
            return "頑健な予測です。NOE寄与は複数の配座または原子ペアに分散しています。"
        if largest_contribution_percent < 50:
            return "中程度に頑健な予測です。NOE寄与はやや集中していますが、単一寄与に支配されてはいません。"
        if largest_contribution_percent < 80:
            return "エネルギー分布に敏感な予測です。NOE寄与は限られた配座または原子ペアに強く依存しています。"
        return "非常に敏感な予測です。NOE寄与がほぼ単一の配座または原子ペアに支配されており、慎重に解釈する必要があります。"

    if largest_contribution_percent < 20:
        return "Robust prediction: the NOE contribution is distributed over several conformers/atom pairs."
    if largest_contribution_percent < 50:
        return "Moderately robust prediction: the NOE contribution is somewhat concentrated but not dominated by a single contribution."
    if largest_contribution_percent < 80:
        return "Energy-sensitive prediction: the NOE contribution is strongly dependent on a limited number of conformers/atom pairs."
    return "Highly sensitive prediction: the NOE contribution is dominated by a single conformer/atom pair and should be interpreted cautiously."


def interpret_strictness_sensitivity(score_range, distance_range, language="English"):
    if language == "日本語":
        if score_range < 10 and distance_range < 0.20:
            return "エネルギー厳格度に対して頑健です。配座存在比の平滑化に対して予測は比較的安定しています。"
        if score_range < 25 and distance_range < 0.50:
            return "エネルギー厳格度に対して中程度に敏感です。配座存在比の仮定により予測がやや変化します。"
        return "エネルギー厳格度に敏感です。配座間エネルギー差をどの程度重視するかによって予測が大きく変化します。"

    if score_range < 10 and distance_range < 0.20:
        return "Robust to energy strictness: the prediction is relatively insensitive to conformer population smoothing."
    if score_range < 25 and distance_range < 0.50:
        return "Moderately sensitive to energy strictness: the prediction changes somewhat with population assumptions."
    return "Sensitive to energy strictness: the prediction depends strongly on how conformer energy differences are weighted."


# =========================
# NOE calculation
# =========================

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
    language="English",
):
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

            validate_atom_group(group_A_idx, mol, "Proton A group", pair_name, mol_name)
            validate_atom_group(group_B_idx, mol, "Proton B group", pair_name, mol_name)

            group_r6_sum = 0.0
            group_score_sum = 0.0
            distances = []
            pairwise_contributions = []

            for a_input, a_idx in zip(group_A_input, group_A_idx):
                for b_input, b_idx in zip(group_B_input, group_B_idx):
                    if a_idx == b_idx:
                        continue

                    distance = rdMolTransforms.GetBondLength(conf, a_idx, b_idx)
                    r_minus_6 = distance ** -6 if distance > 0 else 0.0
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
                    pairwise_row.update(props.copy())
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

            detailed_row = {
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

            detailed_row.update(props.copy())
            detailed_rows.append(detailed_row)

    detailed_df = pd.DataFrame(detailed_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)

    summary_rows = []

    if detailed_df.empty:
        return pd.DataFrame(), detailed_df, pairwise_df

    for pair_name, group in detailed_df.groupby("pair_name"):
        weighted_r6_sum = group["population_weighted_group_r_minus_6_sum"].sum()
        weighted_r6_average = group["population_weighted_group_r_minus_6_average"].sum()
        weighted_score_average = group["population_weighted_group_score_average"].sum()

        effective_distance = weighted_r6_sum ** (-1 / 6) if weighted_r6_sum > 0 else None
        normalized_effective_distance = (
            weighted_r6_average ** (-1 / 6) if weighted_r6_average > 0 else None
        )

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

            total_pairwise_weighted_r6 = pairwise_group["population_weighted_r_minus_6"].sum()

            largest_contribution_percent = (
                largest_contribution_row["population_weighted_r_minus_6"]
                / total_pairwise_weighted_r6
                * 100
                if total_pairwise_weighted_r6 > 0
                else None
            )

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

        p_le_25 = group.loc[
            group["conformer_normalized_effective_distance_A"] <= 2.5,
            "population_percent",
        ].sum()
        p_le_30 = group.loc[
            group["conformer_normalized_effective_distance_A"] <= 3.0,
            "population_percent",
        ].sum()
        p_le_35 = group.loc[
            group["conformer_normalized_effective_distance_A"] <= 3.5,
            "population_percent",
        ].sum()
        p_le_40 = group.loc[
            group["conformer_normalized_effective_distance_A"] <= 4.0,
            "population_percent",
        ].sum()

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
                "contact_population_le_2_5_A_percent": p_le_25,
                "contact_population_le_3_0_A_percent": p_le_30,
                "contact_population_le_3_5_A_percent": p_le_35,
                "contact_population_le_4_0_A_percent": p_le_40,
                "minimum_H_H_distance_A": minimum_distance,
                "closest_atom_pair": closest_atom_pair,
                "closest_conformer": closest_conformer,
                "largest_NOE_contribution_atom_pair": largest_contribution_atom_pair,
                "largest_NOE_contribution_conformer": largest_contribution_conformer,
                "largest_NOE_contribution_percent_of_total_r6": largest_contribution_percent,
                "robustness_comment": interpret_robustness(largest_contribution_percent, language=language),
                "contact_population_comment": interpret_contact_population(p_le_30, p_le_35, p_le_40, language=language),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    return summary_df, detailed_df, pairwise_df


def run_strictness_sensitivity(
    mols,
    pair_definitions,
    numbering_mode,
    relative_energies_kcal,
    energy_values,
    temperature,
    energy_window,
    language="English",
):
    strictness_values = [1.0, 2.0, 5.0]
    rows = []

    for strictness in strictness_values:
        populations = calculate_boltzmann_populations(
            relative_energies_kcal=relative_energies_kcal,
            temperature=temperature,
            strictness_factor=strictness,
            energy_window=energy_window,
        )

        summary_df, _, _ = calculate_noe_for_group_pairs(
            mols=mols,
            pair_definitions=pair_definitions,
            numbering_mode=numbering_mode,
            populations=populations,
            relative_energies_kcal=relative_energies_kcal,
            energy_values=energy_values,
            include_props=False,
            language=language,
        )

        for _, row in summary_df.iterrows():
            rows.append(
                {
                    "pair_name": row["pair_name"],
                    "strictness_factor": strictness,
                    "normalized_effective_NOE_distance_A": row["normalized_effective_NOE_distance_A"],
                    "NOE_likelihood_score_percent": row["NOE_likelihood_score_percent"],
                    "prediction": row["prediction"],
                    "contact_population_le_3_0_A_percent": row["contact_population_le_3_0_A_percent"],
                    "contact_population_le_3_5_A_percent": row["contact_population_le_3_5_A_percent"],
                    "largest_NOE_contribution_percent_of_total_r6": row["largest_NOE_contribution_percent_of_total_r6"],
                }
            )

    sensitivity_df = pd.DataFrame(rows)
    summary_rows = []

    if sensitivity_df.empty:
        return sensitivity_df, pd.DataFrame()

    for pair_name, group in sensitivity_df.groupby("pair_name"):
        score_range = (
            group["NOE_likelihood_score_percent"].max()
            - group["NOE_likelihood_score_percent"].min()
        )
        distance_range = (
            group["normalized_effective_NOE_distance_A"].max()
            - group["normalized_effective_NOE_distance_A"].min()
        )

        summary_rows.append(
            {
                "pair_name": pair_name,
                "score_range_percent_points": score_range,
                "normalized_distance_range_A": distance_range,
                "strictness_sensitivity_comment": interpret_strictness_sensitivity(
                    score_range,
                    distance_range,
                    language=language,
                ),
            }
        )

    sensitivity_summary_df = pd.DataFrame(summary_rows)

    return sensitivity_df, sensitivity_summary_df


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

language = st.sidebar.selectbox(
    "Language / 言語",
    ["English", "日本語"],
    index=0,
)

T = LANG[language]

st.title(T["app_title"])
st.caption(APP_VERSION)
st.write(T["app_description"])


# =========================
# Sidebar
# =========================

st.sidebar.header(T["input_settings"])

numbering_mode = st.sidebar.radio(
    T["atom_numbering"],
    ["1-based atom numbers", "0-based RDKit atom indices"],
    index=0,
)

sanitize = st.sidebar.checkbox(
    T["sanitize"],
    value=False,
)

include_props = st.sidebar.checkbox(
    T["include_props"],
    value=True,
)

show_atom_preview = st.sidebar.checkbox(
    T["show_atom_preview"],
    value=True,
)

show_3d_viewer = st.sidebar.checkbox(
    T["show_3d_viewer"],
    value=True,
)

label_mode = st.sidebar.radio(
    T["label_mode"],
    ["H atoms only", "All atoms", "No labels"],
    index=0,
)

viewer_height = st.sidebar.slider(
    T["viewer_height"],
    min_value=350,
    max_value=800,
    value=520,
    step=50,
)

st.sidebar.header(T["boltzmann_settings"])

temperature = st.sidebar.number_input(
    T["temperature"],
    min_value=1.0,
    max_value=5000.0,
    value=298.15,
    step=10.0,
)

strictness_factor = st.sidebar.slider(
    T["strictness_factor"],
    min_value=1.0,
    max_value=10.0,
    value=1.0,
    step=0.5,
)

energy_window = st.sidebar.number_input(
    T["energy_window"],
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.5,
)

st.sidebar.header(T["advanced_analysis"])

run_sensitivity = st.sidebar.checkbox(
    T["run_sensitivity"],
    value=True,
)


# =========================
# File upload
# =========================

uploaded_file = st.file_uploader(
    T["upload_sdf"],
    type=["sdf"],
)

if not uploaded_file:
    st.info(T["upload_prompt"])
    st.stop()

mols = read_sdf(uploaded_file, sanitize=sanitize)

if len(mols) == 0:
    st.error(T["no_valid_mols"])
    st.stop()

st.success(T["read_success"].format(n=len(mols)))

first_mol = mols[0]


# =========================
# Atom numbering helper
# =========================

st.header(T["atom_helper"])

if show_3d_viewer:
    with st.expander(T["viewer_title"], expanded=True):
        st.write(T["viewer_help"])
        render_3dmol_viewer(
            mol=first_mol,
            numbering_mode=numbering_mode,
            label_mode=label_mode,
            viewer_height=viewer_height,
        )
        st.caption(T["viewer_tip"])

if show_atom_preview:
    with st.expander(T["atom_table_title"], expanded=False):
        st.write(T["atom_table_help"])
        atom_df = make_atom_index_preview(first_mol)
        st.dataframe(atom_df, use_container_width=True)


# =========================
# Energy property selection
# =========================

st.header(T["energy_info"])

available_props = get_available_sdf_properties(mols)

if not available_props:
    st.error(T["no_energy_props"])
    st.stop()

preferred_energy_props = [
    "TOTAL_GIBBS_FREE_ENERGY_KCAL/MOL",
    "POTENTIAL_ENERGY_KCAL/MOL",
    "SCF_ENERGY_HARTREE",
    "SCF_ENERGY",
    "GIBBS_FREE_ENERGY",
]

default_energy_index = 0
for preferred in preferred_energy_props:
    if preferred in available_props:
        default_energy_index = available_props.index(preferred)
        break

energy_property = st.selectbox(
    T["energy_property"],
    available_props,
    index=default_energy_index,
)

if "POPULATION" in energy_property.upper() or "BOLTZMANN" in energy_property.upper():
    st.error(T["population_selected_error"])
    st.stop()

energy_property_upper = energy_property.upper()

if "KCAL" in energy_property_upper:
    default_unit_index = 1
elif "KJ" in energy_property_upper:
    default_unit_index = 2
else:
    default_unit_index = 0

energy_unit = st.radio(
    T["energy_unit"],
    ["Hartree", "kcal/mol", "kJ/mol"],
    index=default_unit_index,
    horizontal=True,
)

energy_values = [
    extract_energy_from_property(mol, energy_property)
    for mol in mols
]

if any(x is None for x in energy_values):
    st.error(T["invalid_energy_values"])
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

with st.expander(T["boltzmann_table"], expanded=True):
    st.dataframe(energy_df, use_container_width=True)

    st.download_button(
        label=T["download_boltzmann"],
        data=convert_df_to_csv_bytes(energy_df),
        file_name="boltzmann_populations.csv",
        mime="text/csv",
    )


# =========================
# GUI input for NOE pairs
# =========================

st.header(T["define_pairs"])
st.write(T["define_pairs_help"])

pair_name = st.text_input(
    T["pair_name"],
    value=f"NOE_{len(st.session_state.noe_pairs) + 1}",
)

colA, colB = st.columns(2)

with colA:
    st.subheader(T["proton_a"])

    proton_A = st.number_input(
        T["proton_a_number"],
        min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
        step=1,
        value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
        key="proton_A_main",
    )

    if st.button(T["eq_a"]):
        st.session_state.eq_A_count += 1
        st.rerun()

    eq_A_atoms = []
    for i in range(st.session_state.eq_A_count):
        eq_A = st.number_input(
            T["eq_a_label"].format(i=i + 1),
            min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
            step=1,
            value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
            key=f"eq_A_{i}",
        )
        eq_A_atoms.append(eq_A)

with colB:
    st.subheader(T["proton_b"])

    proton_B = st.number_input(
        T["proton_b_number"],
        min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
        step=1,
        value=1 if numbering_mode == "0-based RDKit atom indices" else 2,
        key="proton_B_main",
    )

    if st.button(T["eq_b"]):
        st.session_state.eq_B_count += 1
        st.rerun()

    eq_B_atoms = []
    for i in range(st.session_state.eq_B_count):
        eq_B = st.number_input(
            T["eq_b_label"].format(i=i + 1),
            min_value=0 if numbering_mode == "0-based RDKit atom indices" else 1,
            step=1,
            value=1 if numbering_mode == "0-based RDKit atom indices" else 2,
            key=f"eq_B_{i}",
        )
        eq_B_atoms.append(eq_B)

group_A_atoms = [proton_A] + eq_A_atoms
group_B_atoms = [proton_B] + eq_B_atoms

st.write(T["current_pair"])

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

col_add, col_clear_eq = st.columns(2)

with col_add:
    if st.button(T["add_pair"]):
        if not pair_name.strip():
            st.warning(T["enter_pair_name_warning"])
        elif len(set(group_A_atoms)) != len(group_A_atoms):
            st.warning(T["duplicate_a_warning"])
        elif len(set(group_B_atoms)) != len(group_B_atoms):
            st.warning(T["duplicate_b_warning"])
        elif set(group_A_atoms) & set(group_B_atoms):
            st.warning(T["overlap_warning"])
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
    if st.button(T["clear_eq"]):
        reset_current_equivalent_fields()
        st.rerun()

if st.session_state.noe_pairs:
    st.subheader(T["added_pairs"])

    pair_table = pd.DataFrame(
        [
            {
                "index": i,
                "pair_name": p["name"],
                "proton_A_group": ",".join(map(str, p["group_A_atoms"])),
                "proton_B_group": ",".join(map(str, p["group_B_atoms"])),
                "number_of_atom_pairs": len(p["group_A_atoms"]) * len(p["group_B_atoms"]),
            }
            for i, p in enumerate(st.session_state.noe_pairs)
        ]
    )

    st.dataframe(pair_table.drop(columns=["index"]), use_container_width=True)

    st.markdown(f"**{T['delete_selected_title']}**")

    pair_options = [
        f"{i + 1}. {p['name']} | A: {','.join(map(str, p['group_A_atoms']))} | B: {','.join(map(str, p['group_B_atoms']))}"
        for i, p in enumerate(st.session_state.noe_pairs)
    ]

    selected_pair_label = st.selectbox(
        T["select_pair_delete"],
        pair_options,
        key="selected_pair_to_delete",
    )

    selected_pair_index = pair_options.index(selected_pair_label)

    col_delete_one, col_delete_all = st.columns(2)

    with col_delete_one:
        if st.button(T["delete_selected"]):
            deleted_pair = st.session_state.noe_pairs.pop(selected_pair_index)
            st.success(T["deleted_pair"].format(name=deleted_pair["name"]))
            st.rerun()

    with col_delete_all:
        if st.button(T["clear_all"]):
            st.session_state.noe_pairs = []
            reset_current_equivalent_fields()
            st.rerun()

pair_definitions = st.session_state.noe_pairs

if not pair_definitions:
    st.info(T["add_one_pair"])
    st.stop()


# =========================
# NOE calculation
# =========================

st.header(T["prediction"])

try:
    summary_df, detailed_df, pairwise_df = calculate_noe_for_group_pairs(
        mols=mols,
        pair_definitions=pair_definitions,
        numbering_mode=numbering_mode,
        populations=populations,
        relative_energies_kcal=relative_energies_kcal,
        energy_values=energy_values,
        include_props=include_props,
        language=language,
    )
except Exception as e:
    st.error(T["calc_error"].format(e=e))
    st.stop()

if run_sensitivity:
    try:
        sensitivity_df, sensitivity_summary_df = run_strictness_sensitivity(
            mols=mols,
            pair_definitions=pair_definitions,
            numbering_mode=numbering_mode,
            relative_energies_kcal=relative_energies_kcal,
            energy_values=energy_values,
            temperature=temperature,
            energy_window=energy_window,
            language=language,
        )
    except Exception as e:
        st.warning(T["sensitivity_failed"].format(e=e))
        sensitivity_df = pd.DataFrame()
        sensitivity_summary_df = pd.DataFrame()
else:
    sensitivity_df = pd.DataFrame()
    sensitivity_summary_df = pd.DataFrame()


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        T["summary"],
        T["conformer_table"],
        T["atom_pair_details"],
        T["wide_tables"],
        T["plots"],
        T["strictness_sensitivity"],
    ]
)

with tab1:
    st.subheader(T["summary_title"])
    st.dataframe(summary_df, use_container_width=True)

    st.download_button(
        label=T["download_summary"],
        data=convert_df_to_csv_bytes(summary_df),
        file_name="noe_prediction_summary.csv",
        mime="text/csv",
    )

    st.subheader(T["intuitive_interpretation"])

    for _, row in summary_df.iterrows():
        pair_name = row["pair_name"]
        score = row["NOE_likelihood_score_percent"]
        normalized_distance = row["normalized_effective_NOE_distance_A"]
        total_distance = row["effective_NOE_distance_A_total_r6"]
        min_distance = row["minimum_H_H_distance_A"]

        interpretation = interpret_noe_likelihood(
            score_percent=score,
            normalized_distance=normalized_distance,
            language=language,
        )

        st.markdown("---")
        st.markdown(f"### {pair_name}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(T["noe_score"], f"{score:.1f}%")

        with col2:
            if pd.notna(normalized_distance):
                st.metric(T["norm_dist"], f"{normalized_distance:.2f} Å")
            else:
                st.metric(T["norm_dist"], "N/A")

        with col3:
            if pd.notna(min_distance):
                st.metric(T["min_dist"], f"{min_distance:.2f} Å")
            else:
                st.metric(T["min_dist"], "N/A")

        show_interpretation_box(interpretation)

        st.markdown(f"#### {T['contact_population']}")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("≤2.5 Å", f"{row['contact_population_le_2_5_A_percent']:.1f}%")
        with c2:
            st.metric("≤3.0 Å", f"{row['contact_population_le_3_0_A_percent']:.1f}%")
        with c3:
            st.metric("≤3.5 Å", f"{row['contact_population_le_3_5_A_percent']:.1f}%")
        with c4:
            st.metric("≤4.0 Å", f"{row['contact_population_le_4_0_A_percent']:.1f}%")

        st.info(row["contact_population_comment"])

        st.markdown(f"#### {T['robustness_index']}")

        robustness_value = row["largest_NOE_contribution_percent_of_total_r6"]
        if pd.notna(robustness_value):
            st.metric(T["largest_r6"], f"{robustness_value:.1f}%")
        else:
            st.metric(T["largest_r6"], "N/A")

        st.write(row["robustness_comment"])

        if run_sensitivity and not sensitivity_summary_df.empty:
            sens_row = sensitivity_summary_df[
                sensitivity_summary_df["pair_name"] == pair_name
            ]

            if not sens_row.empty:
                sens_row = sens_row.iloc[0]

                st.markdown(f"#### {T['strictness_sensitivity']}")

                s1, s2 = st.columns(2)
                with s1:
                    st.metric(T["score_range"], f"{sens_row['score_range_percent_points']:.1f} points")
                with s2:
                    st.metric(T["distance_range"], f"{sens_row['normalized_distance_range_A']:.2f} Å")

                st.warning(sens_row["strictness_sensitivity_comment"])

        if pd.notna(total_distance):
            if language == "日本語":
                st.caption(
                    f"total r⁻⁶実効距離: {total_distance:.2f} Å。"
                    "等価プロトンを含む場合、通常の単一H···H距離との比較には規格化実効距離の方が解釈しやすいです。"
                )
            else:
                st.caption(
                    f"Total r⁻⁶ effective distance: {total_distance:.2f} Å. "
                    "The normalized effective distance is usually easier to compare with a single H···H distance "
                    "when equivalent protons are included."
                )

        if language == "日本語":
            st.caption("NOE観測可能性スコアは、距離と配座存在比に基づく経験的指標であり、真の観測確率ではありません。")
        else:
            st.caption(
                "The NOE likelihood score is an empirical distance- and population-based indicator, "
                "not a true probability of NOE observation."
            )

    st.markdown(f"**{T['interpretation_notes']}**")
    if language == "日本語":
        st.markdown(
            """
            - `contact_population_le_3_0_A_percent` は、配座ごとの規格化実効H···H距離が3.0 Å以下となる配座のBoltzmann存在比を示します。
            - `largest_NOE_contribution_percent_of_total_r6` は、NOE予測が単一の配座または原子ペアにどの程度支配されているかを示します。
            - `strictness sensitivity` は、エネルギー厳格度係数 1.0、2.0、5.0 で結果を比較します。
            """
        )
    else:
        st.markdown(
            """
            - `contact_population_le_3_0_A_percent` indicates the Boltzmann population of conformers in which the conformer-level normalized effective H···H distance is ≤3.0 Å.
            - `largest_NOE_contribution_percent_of_total_r6` indicates whether the NOE prediction is dominated by a single conformer/atom-pair contribution.
            - `strictness sensitivity` compares results under energy strictness factors 1.0, 2.0, and 5.0.
            """
        )

with tab2:
    st.subheader(T["conformer_table_title"])
    st.dataframe(detailed_df, use_container_width=True)

    st.download_button(
        label=T["download_conformer"],
        data=convert_df_to_csv_bytes(detailed_df),
        file_name="noe_conformer_level_table.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader(T["atom_pair_details_title"])
    st.write(T["atom_pair_details_help"])
    st.dataframe(pairwise_df, use_container_width=True)

    st.download_button(
        label=T["download_pairwise"],
        data=convert_df_to_csv_bytes(pairwise_df),
        file_name="noe_atom_pair_details.csv",
        mime="text/csv",
    )

with tab4:
    st.subheader(T["wide_tables_title"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{T['min_distance_wide']}**")
        distance_wide = make_wide_table(
            detailed_df,
            "minimum_H_H_distance_A_in_group",
        )
        st.dataframe(distance_wide, use_container_width=True)

        st.download_button(
            label=T["download_distance_wide"],
            data=convert_df_to_csv_bytes(distance_wide),
            file_name="noe_group_min_distances_wide.csv",
            mime="text/csv",
        )

    with col2:
        st.markdown(f"**{T['r6_wide']}**")
        r6_wide = make_wide_table(
            detailed_df,
            "population_weighted_group_r_minus_6_sum",
        )
        st.dataframe(r6_wide, use_container_width=True)

        st.download_button(
            label=T["download_r6_wide"],
            data=convert_df_to_csv_bytes(r6_wide),
            file_name="noe_group_r6_contributions_wide.csv",
            mime="text/csv",
        )

with tab5:
    st.subheader(T["plots_title"])

    if not detailed_df.empty:
        st.markdown(f"**{T['plot_min_distance']}**")

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

        st.markdown(f"**{T['plot_r6']}**")

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
                    alt.Tooltip("population_weighted_group_r_minus_6_sum:Q", format=".6g"),
                    alt.Tooltip("population_percent:Q", format=".2f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(r6_chart, use_container_width=True)

    if not pairwise_df.empty:
        st.markdown(f"**{T['plot_pairwise']}**")

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

with tab6:
    st.subheader(T["sensitivity_title"])

    if not run_sensitivity:
        st.info(T["sensitivity_off"])
    elif sensitivity_df.empty:
        st.warning(T["sensitivity_none"])
    else:
        st.write(T["sensitivity_help"])

        st.markdown(f"**{T['sensitivity_summary']}**")
        st.dataframe(sensitivity_summary_df, use_container_width=True)

        st.download_button(
            label=T["download_sensitivity_summary"],
            data=convert_df_to_csv_bytes(sensitivity_summary_df),
            file_name="noe_strictness_sensitivity_summary.csv",
            mime="text/csv",
        )

        st.markdown(f"**{T['sensitivity_details']}**")
        st.dataframe(sensitivity_df, use_container_width=True)

        st.download_button(
            label=T["download_sensitivity_details"],
            data=convert_df_to_csv_bytes(sensitivity_df),
            file_name="noe_strictness_sensitivity_details.csv",
            mime="text/csv",
        )

        score_chart = (
            alt.Chart(sensitivity_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("strictness_factor:Q", title="Energy strictness factor"),
                y=alt.Y(
                    "NOE_likelihood_score_percent:Q",
                    title="NOE likelihood score / %",
                ),
                color=alt.Color("pair_name:N", title="NOE pair"),
                tooltip=[
                    "pair_name",
                    "strictness_factor",
                    alt.Tooltip("NOE_likelihood_score_percent:Q", format=".2f"),
                    alt.Tooltip("normalized_effective_NOE_distance_A:Q", format=".3f"),
                    "prediction",
                ],
            )
            .interactive()
        )

        st.altair_chart(score_chart, use_container_width=True)

        distance_chart = (
            alt.Chart(sensitivity_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("strictness_factor:Q", title="Energy strictness factor"),
                y=alt.Y(
                    "normalized_effective_NOE_distance_A:Q",
                    title="Normalized effective distance / Å",
                ),
                color=alt.Color("pair_name:N", title="NOE pair"),
                tooltip=[
                    "pair_name",
                    "strictness_factor",
                    alt.Tooltip("normalized_effective_NOE_distance_A:Q", format=".3f"),
                    alt.Tooltip("NOE_likelihood_score_percent:Q", format=".2f"),
                    "prediction",
                ],
            )
            .interactive()
        )

        st.altair_chart(distance_chart, use_container_width=True)

st.divider()
st.caption(T["caption"])
