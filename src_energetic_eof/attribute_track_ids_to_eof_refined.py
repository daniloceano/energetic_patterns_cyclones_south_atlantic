import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar os dados das PCs com track_ids
pcs_path = "../csv_eofs_energetics_with_track/Total/pcs.csv"  # Adapte para sua fase ou Total
pcs_df = pd.read_csv(pcs_path)

# Selecionar apenas as colunas das PCs
pcs_columns = [col for col in pcs_df.columns if col.startswith("PC")]

# Criar uma cópia para armazenar os ciclones filtrados
pcs_filtered = pcs_df.copy()
pcs_filtered["dominant_eof"] = None  # Inicializar com None

# Aplicar o critério: a PC da EOF alvo deve ser positiva e 2x maior que as demais PCs em módulo
for pc in pcs_columns:
    pc_idx = int(pc.replace("PC", ""))  # Extrai o número da EOF correspondente
    
    # Máscara: PC deve ser positiva e pelo menos 2x maior que o módulo das outras PCs
    mask = (pcs_df[pc] > 0) & (pcs_df[pc] >= 2 * pcs_df[pcs_columns].drop(columns=[pc]).abs().max(axis=1))
    
    # Atribuir EOF dominante apenas para os ciclones que atendem ao critério
    pcs_filtered.loc[mask, "dominant_eof"] = pc_idx

# Filtrar apenas EOFs 1 a 4
pcs_filtered = pcs_filtered[pcs_filtered["dominant_eof"].isin([1, 2, 3, 4])]

# Contagem de ciclones atribuídos a cada EOF
plt.figure(figsize=(8, 6))
sns.countplot(data=pcs_filtered, x="dominant_eof", palette="muted")
plt.xlabel("EOF", fontsize=14)
plt.ylabel("Number of Cyclones", fontsize=14)
plt.title("Number of Cyclones Assigned to Each EOF (Refined Criteria)", fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# Criar um dataframe longo para facilitar a plotagem dos boxplots por EOF (apenas PCs 1 a 4)
pcs_filtered_melted = pcs_filtered.melt(id_vars=["track_id", "dominant_eof"], value_vars=["PC1", "PC2", "PC3", "PC4"], var_name="PC", value_name="Value")

# Boxplot das PCs separadas por EOF predominante
plt.figure(figsize=(12, 6))
sns.boxplot(data=pcs_filtered_melted, x="PC", y="Value", hue="dominant_eof", palette="Set1")
plt.xlabel("Principal Component", fontsize=14)
plt.ylabel("PC Value", fontsize=14)
plt.title("Boxplot of PCs for Each EOF (Refined Criteria)", fontsize=16)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.legend(title="EOF", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# Exibir alguns ciclones por EOF para inspeção, mostrando as PCs
for eof in sorted(pcs_filtered["dominant_eof"].unique()):
    print(f"\nCiclones atribuídos à EOF {eof}:")
    print(pcs_filtered[pcs_filtered["dominant_eof"] == eof][["track_id", "PC1", "PC2", "PC3", "PC4"]].head(10))  # Mostra track_id e PCs relevantes
    print("-" * 80)

# Mostrar o número de ciclones atribuídos a cada EOF
print("\nNumber of Cyclones Assigned to Each EOF (Refined Criteria):")
print(pcs_filtered["dominant_eof"].value_counts())

# # Salvar resultado atualizado
pcs_filtered.to_csv("../csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_refined.csv", index=False)
print("Resultado atualizado salvo em 'pcs_with_dominant_eof_refined.csv'")