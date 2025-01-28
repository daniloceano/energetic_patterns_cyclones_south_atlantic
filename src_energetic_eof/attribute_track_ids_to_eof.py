import pandas as pd
import numpy as np

# Carregar os dados das PCs com track_ids
pcs_path = "../csv_eofs_energetics_with_track/Total/pcs.csv"  # Adapte para sua fase ou Total
pcs_df = pd.read_csv(pcs_path)

# Identificar a EOF dominante para cada ciclone
pcs_columns = [col for col in pcs_df.columns if col.startswith("PC")]
pcs_df["dominant_eof"] = pcs_df[pcs_columns].abs().idxmax(axis=1)  # Coluna com EOF dominante
pcs_df["dominant_eof"] = pcs_df["dominant_eof"].str.extract(r'(\d+)').astype(int)  # Extrair apenas o número da EOF

# Salvar resultado atualizado
pcs_df.to_csv("../csv_eofs_energetics_with_track/Total/pcs_with_dominant_eof.csv", index=False)
