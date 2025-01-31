import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Caminho para os dados das PCs com track_ids
pcs_path = "../csv_eofs_energetics_with_track/Total/pcs.csv"  # Adapte para sua fase ou Total
pcs_df = pd.read_csv(pcs_path)

# Selecionar apenas as colunas das PCs
pcs_columns = [col for col in pcs_df.columns if col.startswith("PC")]

# Calcular os quantis 90 e 10 para cada PC
quantiles_90 = pcs_df[pcs_columns].quantile(0.90)  # Valores positivos extremos
quantiles_10 = pcs_df[pcs_columns].quantile(0.10)  # Valores negativos extremos

# **Atribuição com base no quantil 90% (valores positivos)**
pcs_filtered_q90 = pcs_df[(pcs_df[pcs_columns] >= quantiles_90).any(axis=1)].copy()
pcs_filtered_q90["dominant_eof"] = pcs_filtered_q90[pcs_columns].idxmax(axis=1)  # Pega a PC com o maior valor positivo
pcs_filtered_q90["dominant_eof"] = pcs_filtered_q90["dominant_eof"].str.extract(r'(\d+)').astype(int)  # Extrai o número da EOF
pcs_filtered_q90 = pcs_filtered_q90[pcs_filtered_q90["dominant_eof"] <= 4]  # Manter apenas EOFs de 1 a 4

# **Atribuição com base no quantil 10% (valores negativos)**
pcs_filtered_q10 = pcs_df[(pcs_df[pcs_columns] <= quantiles_10).any(axis=1)].copy()
pcs_filtered_q10["dominant_eof"] = pcs_filtered_q10[pcs_columns].idxmin(axis=1)  # Pega a PC com o menor valor negativo
pcs_filtered_q10["dominant_eof"] = pcs_filtered_q10["dominant_eof"].str.extract(r'(\d+)').astype(int)  # Extrai o número da EOF
pcs_filtered_q10 = pcs_filtered_q10[pcs_filtered_q10["dominant_eof"] <= 4]  # Manter apenas EOFs de 1 a 4

# **Plotando a contagem de ciclones atribuídos para cada EOF**

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico para o critério de quantil 90%
sns.countplot(data=pcs_filtered_q90, x="dominant_eof", palette="muted", ax=axes[0])
axes[0].set_xlabel("EOF", fontsize=14)
axes[0].set_ylabel("Number of Cyclones", fontsize=14)
axes[0].set_title("Number of Cyclones Assigned to Each EOF (90th Percentile)", fontsize=16)
axes[0].tick_params(axis='x', labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].grid(axis="y", linestyle="--", alpha=0.5)

# Gráfico para o critério de quantil 10%
sns.countplot(data=pcs_filtered_q10, x="dominant_eof", palette="muted", ax=axes[1])
axes[1].set_xlabel("EOF", fontsize=14)
axes[1].set_ylabel("Number of Cyclones", fontsize=14)
axes[1].set_title("Number of Cyclones Assigned to Each EOF (10th Percentile)", fontsize=16)
axes[1].tick_params(axis='x', labelsize=12)
axes[1].tick_params(axis='y', labelsize=12)
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# **Exibir alguns ciclones atribuídos a cada EOF para inspeção**
for eof in sorted(pcs_filtered_q90["dominant_eof"].unique()):
    print(f"\nCiclones atribuídos à EOF {eof} (Q90 - Valores Positivos):")
    print(pcs_filtered_q90[pcs_filtered_q90["dominant_eof"] == eof][["track_id", "PC1", "PC2", "PC3", "PC4"]].head(10))
    print("-" * 80)

for eof in sorted(pcs_filtered_q10["dominant_eof"].unique()):
    print(f"\nCiclones atribuídos à EOF {eof} (Q10 - Valores Negativos):")
    print(pcs_filtered_q10[pcs_filtered_q10["dominant_eof"] == eof][["track_id", "PC1", "PC2", "PC3", "PC4"]].head(10))
    print("-" * 80)

# **Salvar os arquivos atualizados**
q90_output_path = "../csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q90.csv"
q10_output_path = "../csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof_q10.csv"

pcs_filtered_q90.to_csv(q90_output_path, index=False)
pcs_filtered_q10.to_csv(q10_output_path, index=False)

print(f"Arquivo salvo: {q90_output_path}")
print(f"Arquivo salvo: {q10_output_path}")
