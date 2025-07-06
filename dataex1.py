import re
import sys,time
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

file_path = 'newdata.csv'
# file_path = 'C:\Users\Vignesh\Documents\da_ex\datalsCopy.csv'

# df = pd.read_csv(file_path)
df = pd.read_csv(file_path, encoding='utf-8')

# --- 1. Extract Data Function ---

def convert_to_j_per_cm3(item):
    j_per_cm3 = []
    values = item.split(',')

    for value in values:
        value = value.strip()
        v_ = ''
        if not re.search(r'\d', value):
            # j_per_cm3.append('')
            continue
        if re.match(r'\d+\.?\d*%\s*J/cm\(3\)', value):
            match = re.search(r'(\d+\.?\d*)%\s*J/cm\(3\)', value)
            if match:
                percentage = float(match.group(1)) / 100
                v_ = f"{percentage * 1.20:.2f} J/cm³"

        elif re.match(r'\d+\.?\d*\s*J', value):
            match = re.search(r'(\d+\.?\d*)\s*J', value)
            if match:
                v_ = f"{float(match.group(1)):.2f} J/cm³"
        
        elif re.match(r'<\d+\.?\d*\s*J/cm\(-3\)', value):
            match = re.search(r'<(\d+\.?\d*)\s*J/cm\(-3\)', value)
            if match:
                v_ = f"<{float(match.group(1)):.2f} J/cm³"

        elif re.match(r'\d+\.?\d*\s*j/cm', value):
            match = re.search(r'(\d+\.?\d*)\s*j/cm', value)
            if match:
                v_ = f"{float(match.group(1)):.2f} J/cm³"
        elif re.match(r'\d+\.?\d*\s*mJ/cm\(-3\)', value):
            match = re.search(r'(\d+\.?\d*)\s*mJ/cm\(-3\)', value)
            if match:
                v_ = f"{match.group(1)} mJ/cm³"
        elif  re.match(r'\d+\.?\d*\s*J\s*cm\(-1\)', value):
            match = re.search(r'(\d+\.?\d*)\s*J\s*cm\(-1\)', value)
            if match:
                v_ = f"{float(match.group(1)):.2f} J/cm³"  
        else:
            if re.match(r'= \d*.\d* J/cm',value):
                match = re.search(r'= (\d*.\d*)',value)
                if match:
                    v_ = f"{float(match.group(1)):.2f} J/cm³"
            v_ = value 

        if '-' in str(v_):
            if re.match(r'\d*-\d* J',v_):
                if re.match(r'-\d* J/cm3',v_):
                    v_ = str(v_).replace('-','')
                if re.match(r'-\d* J cm',v_):
                    v_ = str(v_).replace('-','')
                if re.match(r'\d*-\d* J/cm\(3\)',v_):
                    v_ = f"{re.search(r'\d*',v_).group()} J/cm³"
                match = re.search(r'(\d*)-(\d*)\s*J/cm', v_)
                if match:
                    v_ = f"{re.search(r'\d*',match.group()).group()} J/cm³"
                elif re.match(r'(\d*)-(\d*)\s*J cm(-3)', v_):
                    match = re.search(r'(\d*)-(\d*)\s*J cm(-3)', v_)
                    v_ = f"{re.search(r'\d*',match.group()).group()} J/cm³"
                else:
                    match = re.search(r'(\d*)-(\d*)\s*J', v_)
                    v_ = f"{re.search(r'\d*',match.group()).group()} J/cm³"
            else:
                v_ = str(v_).replace('-','') 
        j_per_cm3.append(v_)
    if j_per_cm3:
        return ','.join(j_per_cm3)
    else: return ''

def extract_data(index_id, abstract):
    global index_id_,abstract_
    index_id_ = index_id
    abstract_ = abstract
    lead = False
    data = {
        'Index': index_id,
        'energy_storage_density': '',
        'Converted_energy_storage_density': '',
        'No_of_energy_storage_density': '',
        'material_composition': '',
        'reason_for_high_energy_storage_density': '',
        'thin_film_or_bulk': ''
    }
    # m = [r'([A-Za-z0-9\(\)\-]+(?:O3?|\d+)[A-Za-z0-9\(\)\-]*)']
    formula_pattern = r'([A-Z][a-z]?\d*(?:[-.,]?\d+)?(?:\([^\)]*\))?[\d\w\-\/]*|[A-Za-z]+(?:\/[A-Za-z]+)?(?:\d*\.?\d+))'
    formulas = re.findall(formula_pattern, abstract)
    filtered_formulas = [f for f in formulas if 'O' in f or 'KNb' in f or 'Ba' in f or 'Ti' in f or 'Zr' in f or 'Na' in f or 'Bi' in f or 'Sr' in f]
    data['material_composition'] = ','.join(list(set(filtered_formulas)))
    if 'PB' in data['material_composition'] or 'pb' in data['material_composition'] or 'Pb' in data['material_composition']or 'Pb0' in str(data['material_composition']):
        lead = True

    energy_patterns = [  
        r'W-rec of \d*.\d* J/cm\(\d*\)|-\d*.\d* J/cm\(\d*\)|-\d*.\d* J/cm\d+| \d* J/cm\(\d*\)|\d* J/cm\(\d*\)|\d*.\d* J/cm\(\d\)|\d*.\d* J\/cm3|\d*.\d J cm\(.*?\)|\d.*\d*J/cm\(.*?\)\d*.|d* Jcm\(.*?\)|\d*.\d* j/cm|\d*.\d* J cm-1|\d*.\d* J cm|\d*.\d* J',  
    ]
    a = []
    for pattern in energy_patterns:
        match = re.findall(pattern, abstract)
        if match:
            if match not in a:a.append(','.join(list(set(match))))
    
    dd  = ','.join(list(set(a))).strip()

    if len(dd) > 50:
        a = re.findall(r'\d* J/cm\(\d*\)|\d*.\d* J/cm\(\d*\)|\d*.\d*J/cm\(\d*\)|\d*.\d* J/cm\(.*?\)\w|\d* J/cm\(.*?\)',str(dd))
        data['energy_storage_density'] = ','.join(list(set(a)))
    else:
        data['energy_storage_density'] =dd
    
    data['Converted_energy_storage_density'] = convert_to_j_per_cm3( data['energy_storage_density'])

    reason_patterns = [
        r'antiferroelectric|antiferroelectric\s*phase',
        r'(high\s*energy\s*storage\s*density|optimized\s*microstructure|large\s*breakdown\s*strength|increased\s*dielectric\s*constant|high\s*permittivity|polar\s*nanoregions|cationic\s*disorder|ferroelectric\s*destabilization|relaxation|dielectric\s*breakdown\s*strength|low\s*leakage\s*current|increased\s*polarization|low\s*loss\s*factor|improved\s*stability)',
        r'(energy\s*storage\s*performance|enhanced\s*relaxation|dielectric\s*loss\s*reduction|polarization\s*enhancement|structural\s*optimization|local\s*polarization\s*enhancement)'
    ]

    reason_matches = []
    for pattern in reason_patterns:
        match = re.findall(pattern, abstract, re.IGNORECASE)
        if match:
            # reason_matches.append(match.group(0))
            reason_matches.append(match)

    data['reason_for_high_energy_storage_density'] = ','.join([','.join(i) for i in reason_matches])

    thin_keywords = ['thin film', 'thin-film', 'nanofilm', 'thin layer', 'layer', 'film', 'coating', 'monolayer', '2d material',' 2d' 'nanocomposite','nanocomposites','(2d)','polymer-base','robotics' ]
    bulk_keywords = ['bulk material', 'bulk form', 'massive', '3d structure', 'solid material', 'macroscopic material', 'block material', 'full body','ceramics','ceramic','single crystals']

    if any(keyword.lower() in abstract.lower() for keyword in thin_keywords):
        data['thin_film_or_bulk'] = 'Thin film'
    elif any(keyword.lower() in abstract.lower() for keyword in bulk_keywords):
        data['thin_film_or_bulk'] = 'Bulk'
    else:
        data['thin_film_or_bulk'] = ''

    return data,lead


def is_duplicate(data, category_list):
    return any(item['Title'] == data['Title'] for item in category_list)

keywords = {
    "NaBa": [
        "NaBa","Sodium Barium", "Na0.5Ba0.5", "Na-Ba", "Sodium/Barium compound", 
        "Sodium Barium alloy", "NaBa-based materials", "Sodium Barium compounds"
    ],
    "Ba": [
        "Ba", "Barium", "Barium oxide", "BaO", "Barium titanate", "BaTiO3", 
        "Barium sulfate", "BaSO4", "Barium carbonate", "BaCO3", "Barium chloride", "BaCl2"
    ],
    "KNb": [
        "KNb", "Potassium Niobate", "KNbO3", "Potassium Niobate crystals", 
        "Potassium-Niobium compound", "Potassium Niobate-based materials"
    ],
}


NaBa_bluk = []
NaBa_thin = []
NaBa_unknown = []

Ba_bluk = []
Ba_thin = []
Ba_unknown = []

KNb_bluk = []
KNb_thin = []
KNb_unknown = []

lead_bluk = []
lead_thin = []
lead_unknown = []

bluk = []
thin = []
unknown = []

for index, row in df.iterrows():
    # if index < 10: 
        abstract = row['Abstract']
        title = row['Article Title']
        DOI = row['DOI']

        # if DOI  in ['10.1021/acsapm.2c00327','10.1021/acssuschemeng.1c05597','10.1111/jace.15429','10.1063/1.4892454','10.1109/LED.2023.3296945','10.1039/d4tc00818a','10.1016/j.ceramint.2023.09.047','10.1016/j.actamat.2023.119071','10.1007/s11664-019-07370-9','10.1007/s10854-023-11774-z','10.1002/aelm.202400001','10.1039/c9ta13951f','10.4191/kcers.2019.56.4.10','10.1002/adma.202108772','10.26599/JAC.2024.9220920','10.1002/adma.202402070','10.1002/adma.202302554','10.1002/aenm.202200517','10.1039/d0ta08335f','10.1021/acs.jpcc.3c07742','10.1021/acs.jpcc.0c11629','10.1021/acsami.1c20214','10.1111/jace.15371','10.1021/acsami.2c14302','10.1021/acs.jpcc.3c03014','10.1016/j.cej.2024.152365','10.1016/j.apsusc.2021.149992']:
        #     continue
        try:data,lead = extract_data(index, abstract)
        except:continue
        
        if len(str(data['Converted_energy_storage_density']).strip().split(',')) > 1:
            d = str(data['Converted_energy_storage_density']).strip().split(',')
            data['No_of_energy_storage_density'] = len(d)

        data['DOI'] = DOI
        data['Abstract'] = abstract
        data['Title'] = title
        
        if lead:
            if data['thin_film_or_bulk'] == 'Bulk':
                lead_bluk.append(data)
            elif data['thin_film_or_bulk'] == 'Thin film':
                lead_thin.append(data)
            else:
                lead_unknown.append(data)

        elif 'Lead-free' in abstract:
            if data['thin_film_or_bulk'] == 'Bulk':
                lead_bluk.append(data)
            elif data['thin_film_or_bulk'] == 'Thin film':
                lead_thin.append(data)
            else:
                lead_unknown.append(data)
        else:
            for material, key_words in keywords.items():
                if any(keyword in data['material_composition'] or 
                    keyword in data['Abstract']for keyword in key_words):
                    if material == "NaBa":
                        if data['thin_film_or_bulk'] == 'Bulk' and not is_duplicate(data, NaBa_bluk):
                            NaBa_bluk.append(data)
                        elif data['thin_film_or_bulk'] == 'Thin film' and not is_duplicate(data, NaBa_thin):
                            NaBa_thin.append(data)
                        else:
                            if not is_duplicate(data, NaBa_unknown):
                                NaBa_unknown.append(data)

                    elif material == "Ba":
                        if data['thin_film_or_bulk'] == 'Bulk' and not is_duplicate(data, Ba_bluk):
                            Ba_bluk.append(data)
                        elif data['thin_film_or_bulk'] == 'Thin film' and not is_duplicate(data, Ba_thin):
                            Ba_thin.append(data)
                        else:
                            if not is_duplicate(data, Ba_unknown):
                                Ba_unknown.append(data)

                    elif material == "KNb":
                        if data['thin_film_or_bulk'] == 'Bulk' and not is_duplicate(data, KNb_bluk):
                            KNb_bluk.append(data)
                        elif data['thin_film_or_bulk'] == 'Thin film' and not is_duplicate(data, KNb_thin):
                            KNb_thin.append(data)
                        else:
                            if not is_duplicate(data, KNb_unknown):
                                KNb_unknown.append(data)
                    break
            
            else:
                if data['thin_film_or_bulk'] == 'Bulk':
                    bluk.append(data)
                elif data['thin_film_or_bulk'] == 'Thin film':
                    thin.append(data)
                else:
                    unknown.append(data)

# print('Lead --------------------')
# print(len(lead_bluk))
# print(len(lead_thin))
# print(len(lead_unknown))
# print('NaBa --------------------')
# print(len(NaBa_bluk))
# print(len(NaBa_thin))
# print(len(NaBa_unknown))
# print('Ba --------------------')
# print(len(Ba_bluk))
# print(len(Ba_thin))
# print(len(Ba_unknown))
# print('KNB --------------------')
# print(len(KNb_bluk))
# print(len(KNb_thin))
# print(len(KNb_unknown))
# print('Different --------------------')
# print(len(bluk))
# print(len(thin))
# print(len(unknown))
# exit()

def safe_sort(df, column_name='Converted_energy_storage_density'):
    if column_name in df.columns:
        return df.sort_values(by=column_name, ascending=True)
    else:
        # print(f"Warning: Column '{column_name}' not found in DataFrame.")
        return df

df_NaBa_bluk = pd.DataFrame(NaBa_bluk)
df_NaBa_thin = pd.DataFrame(NaBa_thin)
df_NaBa_unknown = pd.DataFrame(NaBa_unknown)

df_NaBa_bluk = safe_sort(df_NaBa_bluk)
df_NaBa_thin = safe_sort(df_NaBa_thin)
df_NaBa_unknown = safe_sort(df_NaBa_unknown)


df_Ba_bluk = pd.DataFrame(Ba_bluk)
df_Ba_thin = pd.DataFrame(Ba_thin)
df_Ba_unknown = pd.DataFrame(Ba_unknown)

df_Ba_bluk = safe_sort(df_Ba_bluk)
df_Ba_thin = safe_sort(df_Ba_thin)
df_Ba_unknown = safe_sort(df_Ba_unknown)

df_KNb_bluk = pd.DataFrame(KNb_bluk)
df_KNb_thin = pd.DataFrame(KNb_thin)
df_KNb_unknown = pd.DataFrame(KNb_unknown)

df_KNb_bluk = safe_sort(df_KNb_bluk)
df_KNb_thin = safe_sort(df_KNb_thin)
df_KNb_unknown = safe_sort(df_KNb_unknown)


df_lead_bluk = pd.DataFrame(lead_bluk)
df_lead_thin = pd.DataFrame(lead_thin)
df_lead_unknown = pd.DataFrame(lead_unknown)

df_lead_bluk = safe_sort(df_lead_bluk)
df_lead_thin = safe_sort(df_lead_thin)
df_lead_unknown = safe_sort(df_lead_unknown)

df_bluk = pd.DataFrame(bluk)
df_thin = pd.DataFrame(thin)
df_unknown = pd.DataFrame(unknown)

df_bluk = safe_sort(df_bluk)
df_thin = safe_sort(df_thin)
df_unknown = safe_sort(df_unknown)

output_file_path = 'categorized_materials.xlsx'
lead_output_file_path = 'leadFree_materials.xlsx'

with pd.ExcelWriter(lead_output_file_path) as writer:
    df_lead_bluk.to_excel(writer, sheet_name='Lead Bulk', index=False)
    df_lead_thin.to_excel(writer, sheet_name='Lead Thin film', index=False)
    df_lead_unknown.to_excel(writer, sheet_name='Lead Unknown', index=False)

with pd.ExcelWriter(output_file_path) as writer:
    df_NaBa_bluk.to_excel(writer, sheet_name='NaBa Bulk', index=False)
    df_NaBa_thin.to_excel(writer, sheet_name='NaBa Thin film', index=False)
    df_NaBa_unknown.to_excel(writer, sheet_name='NaBa Unknown', index=False)
    
    df_Ba_bluk.to_excel(writer, sheet_name='Ba Bulk', index=False)
    df_Ba_thin.to_excel(writer, sheet_name='Ba Thin film', index=False)
    df_Ba_unknown.to_excel(writer, sheet_name='Ba Unknown', index=False)
    
    df_KNb_bluk.to_excel(writer, sheet_name='KNb Bulk', index=False)
    df_KNb_thin.to_excel(writer, sheet_name='KNb Thin film', index=False)
    df_KNb_unknown.to_excel(writer, sheet_name='KNb Unknown', index=False)
    
    df_bluk.to_excel(writer, sheet_name='General Bulk', index=False)
    df_thin.to_excel(writer, sheet_name='General Thin film', index=False)
    df_unknown.to_excel(writer, sheet_name='General Unknown', index=False)
exit()
def convert_energy_storage_density(row):
    if 'Converted_energy_storage_density' in row:
        densities = row['Converted_energy_storage_density'].split(',')
        numeric_densities = []
        for d in densities:
            match = re.match(r'([+-]?\d*\.?\d+)', d.strip())
            if match:
                numeric_densities.append(float(match.group(0)))
        return np.mean(numeric_densities) if numeric_densities else np.nan
    else:
        print("Column 'Converted_energy_storage_density' not found!")
        return np.nan

def apply_energy_storage_density(df, material_type):
    if df.empty:
        # print(f"Warning: {material_type} DataFrame is empty.")
        return df
    if 'Converted_energy_storage_density' not in df.columns:
        # print(f"Warning: 'Converted_energy_storage_density' column is missing in {material_type} DataFrame.")
        return df
    df['Energy Storage Density'] = df.apply(convert_energy_storage_density, axis=1)
    return df

df_NaBa_bluk = apply_energy_storage_density(df_NaBa_bluk, "NaBa Bulk")
df_NaBa_thin = apply_energy_storage_density(df_NaBa_thin, "NaBa Thin Film")
df_NaBa_unknown = apply_energy_storage_density(df_NaBa_unknown, "NaBa Unknown")

df_Ba_bluk = apply_energy_storage_density(df_Ba_bluk, "Ba Bulk")
df_Ba_thin = apply_energy_storage_density(df_Ba_thin, "Ba Thin Film")
df_Ba_unknown = apply_energy_storage_density(df_Ba_unknown, "Ba Unknown")

df_KNb_bluk = apply_energy_storage_density(df_KNb_bluk, "KNb Bulk")
df_KNb_thin = apply_energy_storage_density(df_KNb_thin, "KNb Thin Film")
df_KNb_unknown = apply_energy_storage_density(df_KNb_unknown, "KNb Unknown")

df_bluk = apply_energy_storage_density(df_bluk, "General Bulk")
df_thin = apply_energy_storage_density(df_thin, "General Thin Film")
df_unknown = apply_energy_storage_density(df_unknown, "General Unknown")

def get_cleaned_data(df, material_type):
    """Returns the cleaned (non-NaN) energy storage density data for a specific material type."""
    if 'Energy Storage Density' not in df.columns:
        # print(f"Column 'Energy Storage Density' is missing in {material_type}.")
        return pd.Series()
    return df[df['Energy Storage Density'].notna()]

thin_film_naba = get_cleaned_data(df_NaBa_thin, 'NaBa Thin Film')
bulk_data_naba = get_cleaned_data(df_NaBa_bluk, 'NaBa Bulk')
unknown_data_naba = get_cleaned_data(df_NaBa_unknown, 'NaBa Unknown')

thin_film_ba = get_cleaned_data(df_Ba_thin, 'Ba Thin Film')
bulk_data_ba = get_cleaned_data(df_Ba_bluk, 'Ba Bulk')
unknown_data_ba = get_cleaned_data(df_Ba_unknown, 'Ba Unknown')

thin_film_knb = get_cleaned_data(df_KNb_thin, 'KNb Thin Film')
bulk_data_knb = get_cleaned_data(df_KNb_bluk, 'KNb Bulk')
unknown_data_knb = get_cleaned_data(df_KNb_unknown, 'KNb Unknown')

thin_film_data = get_cleaned_data(df_thin, 'General Thin Film')
bulk_data = get_cleaned_data(df_bluk, 'General Bulk')
unknown_data = get_cleaned_data(df_unknown, 'General Unknown')

df_combined = pd.concat([df_NaBa_bluk, df_NaBa_thin, df_NaBa_unknown,
                         df_Ba_bluk, df_Ba_thin, df_Ba_unknown,
                         df_KNb_bluk, df_KNb_thin, df_KNb_unknown,
                         df_bluk, df_thin, df_unknown], ignore_index=True)

def plot_distribution(data, label, title, filename, color='blue'):
    if data.empty:
        return
    plt.figure(figsize=(10, 6))
    sns.histplot(data, kde=True, color=color, label=label, stat="density")
    plt.title(title)
    plt.xlabel('Energy Storage Density (J/cm³)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


plot_distribution(thin_film_naba['Energy Storage Density'], 'NaBa Thin Film', 'Energy Storage Density Distribution - NaBa Thin Film', 'naba_thinFilm.png')
plot_distribution(thin_film_ba['Energy Storage Density'], 'Ba Thin Film', 'Energy Storage Density Distribution - Ba Thin Film', 'ba_thinFilm.png')
if not thin_film_knb.empty and 'Energy Storage Density' in thin_film_knb.columns:
    plot_distribution(thin_film_knb['Energy Storage Density'], 'KNb Thin Film', 'Energy Storage Density Distribution - KNb Thin Film', 'knb_thinFilm.png')

plot_distribution(bulk_data_naba['Energy Storage Density'],  'Ba Bulk Film', 'Energy Storage Density Distribution - Ba Bluk Film', 'naba_Bulk.png',color='red')
plot_distribution(bulk_data_ba['Energy Storage Density'], 'Ba Bulk', 'Energy Storage Density Distribution - Ba Bulk', 'ba_bulk.png', color='red')
plot_distribution(bulk_data_knb['Energy Storage Density'], 'KNb Bulk', 'Energy Storage Density Distribution - KNb Bulk', 'knb_bulk.png', color='red')

plot_distribution(unknown_data_naba['Energy Storage Density'], 'NaBa Unknown', 'Energy Storage Density Distribution - NaBa Unknown', 'naba_unknown.png', color='green')
plot_distribution(unknown_data_ba['Energy Storage Density'], 'Ba Unknown', 'Energy Storage Density Distribution - Ba Unknown', 'ba_unknown.png', color='green')
if not unknown_data_knb.empty and 'Energy Storage Density' in unknown_data_knb.columns:
    plot_distribution(unknown_data_knb['Energy Storage Density'], 'KNb Unknown', 'Energy Storage Density Distribution - KNb Unknown', 'knb_unknown.png', color='green')

plot_distribution(thin_film_data['Energy Storage Density'],  'Different_material Thin Bulk Film', 'Energy Storage Density Distribution - Diffrent Thin Film', 'different_material_thin.png',color='orange')
plot_distribution(bulk_data_ba['Energy Storage Density'], 'Different_material Bulk', 'Energy Storage Density Distribution - Diffrent Bulk', 'different_material_Bulk.png', color='orange')
plot_distribution(bulk_data_knb['Energy Storage Density'], 'Different_material unknow', 'Energy Storage Density Distribution - Diffrent Unknown', 'different_material_unknow.png', color='orange')

plt.figure(figsize=(8, 6))
sns.boxplot(x='thin_film_or_bulk', y='Energy Storage Density', data=df_combined)
plt.title('Energy Storage Density: Thin Film vs Bulk vs Unknown')
plt.xlabel('Material Type')
plt.ylabel('Energy Storage Density (J/cm³)')
plt.tight_layout()
plt.savefig('energy_storage_density_boxplot.png')
plt.show()

# top_25_thin_film = df_thin.nlargest(25, 'Energy Storage Density')
# top_25_bulk = df_bluk.nlargest(25, 'Energy Storage Density')
# top_25_unknown = df_unknown.nlargest(25, 'Energy Storage Density')

# top_25_combined = pd.concat([top_25_thin_film, top_25_bulk, top_25_unknown], ignore_index=True)

# with pd.ExcelWriter('top_25_combined_samples.xlsx') as writer:
#     top_25_thin_film.to_excel(writer, sheet_name='Thin Film', index=False)
#     top_25_bulk.to_excel(writer, sheet_name='Bulk', index=False)
#     top_25_unknown.to_excel(writer, sheet_name='Unknown', index=False)

# top_25_thin_film = df_thin.nsmallest(25, 'Energy Storage Density')
# top_25_bulk = df_bluk.nsmallest(25, 'Energy Storage Density')
# top_25_unknown = df_unknown.nsmallest(25, 'Energy Storage Density')

# top_25_combined = pd.concat([top_25_thin_film, top_25_bulk, top_25_unknown], ignore_index=True)

# with pd.ExcelWriter('least_25_combined_samples.xlsx') as writer:
#     top_25_thin_film.to_excel(writer, sheet_name='Thin Film', index=False)
#     top_25_bulk.to_excel(writer, sheet_name='Bulk', index=False)
#     top_25_unknown.to_excel(writer, sheet_name='Unknown', index=False)


