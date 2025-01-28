import os
import pandas as pd
import numpy as np

# Caminhos para as pastas
original_path = "../csv_eofs_energetics/"
new_path = "../csv_eofs_energetics_with_track/"
files_to_compare = ["pcs.csv", "reconstructed_data.csv", "eofs.csv", "eofs_with_mean.csv"]

def compare_files(original_file, new_file, file_name):
    """Compara arquivos específicos entre as duas pastas."""
    if not os.path.exists(original_file) or not os.path.exists(new_file):
        print(f"Arquivo {file_name} não encontrado em uma das pastas.")
        return None

    # Carregar os DataFrames
    original_df = pd.read_csv(original_file, header=None)
    new_df = pd.read_csv(new_file)

    # Ajustar os DataFrames para comparação
    if "track_id" in new_df.columns:
        new_df = new_df.drop(columns=["track_id"])  # Remover a coluna track_id do novo arquivo

    # Verificar se as dimensões batem
    if original_df.shape != new_df.shape:
        print(f"Os arquivos {file_name} têm tamanhos diferentes: {original_df.shape} vs {new_df.shape}")
        return None

    # Comparar diferenças nos valores
    absolute_diff = np.abs(original_df.values - new_df.values)
    relative_diff = np.abs((original_df.values - new_df.values) / (np.abs(original_df.values) + 1e-10))  # Evitar divisão por zero

    # Resumo das diferenças
    max_abs_diff = np.max(absolute_diff)
    max_rel_diff = np.max(relative_diff)

    return {
        "file_name": file_name,
        "max_absolute_diff": max_abs_diff,
        "max_relative_diff": max_rel_diff,
        "absolute_diff": absolute_diff,
        "relative_diff": relative_diff,
    }

def compare_folders(original_path, new_path, files_to_compare):
    """Compara arquivos em todas as subpastas das duas pastas."""
    results = []

    for phase in os.listdir(original_path):
        phase_path_original = os.path.join(original_path, phase)
        phase_path_new = os.path.join(new_path, phase)
        
        if os.path.isdir(phase_path_original) and os.path.isdir(phase_path_new):
            print(f"Comparando arquivos para a fase: {phase}")
            for file_name in files_to_compare:
                original_file = os.path.join(phase_path_original, file_name)
                new_file = os.path.join(phase_path_new, file_name)

                result = compare_files(original_file, new_file, file_name)
                if result:
                    results.append({
                        "phase": phase,
                        "file_name": result["file_name"],
                        "max_absolute_diff": result["max_absolute_diff"],
                        "max_relative_diff": result["max_relative_diff"],
                    })
    
    return results

# Executar a comparação
results = compare_folders(original_path, new_path, files_to_compare)

# Exibir resultados resumidos
for result in results:
    print(f"Fase: {result['phase']} | Arquivo: {result['file_name']}")
    print(f"  Máxima diferença absoluta: {result['max_absolute_diff']}")
    print(f"  Máxima diferença relativa: {result['max_relative_diff']}")
    print()
