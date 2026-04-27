"""
K-Reinas con Algoritmo Genético - Runner Principal
===================================================
Ejecuta ambas versiones, genera gráficas y guarda resultados.
"""

import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import time
import random
import numpy as np

# Seed para reproducibilidad parcial
random.seed(42)
np.random.seed(42)

"Resultados y gráficos se guardarán en esta carpeta local (dentro del usuario actual)"
OUTPUT_DIR = "/mnt/k_queens_ga"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Importar módulos locales
from k_queens_genetic import (
    GeneticConfig, GeneticAlgorithm, run_all_boards, calculate_conflicts
)
from k_queens_viz import generate_all_figures

def print_summary_table(results: dict):
    """Imprime tabla resumen en consola"""
    print("\n" + "=" * 80)
    print("  TABLA RESUMEN DE RESULTADOS")
    print("=" * 80)
    print(f"{'Tablero':>10} | {'Versión':>20} | {'Conflictos':>12} | {'Tiempo(s)':>10} | {'Generaciones':>13} | {'Solución':>9}")
    print("-" * 80)

    for n in sorted(results.keys()):
        for ver_key, ver_name in [('v1', 'V1: PMX+Swap'), ('v2', 'V2: OX+Inversión')]:
            stats = results[n][ver_key]['best_stats']
            sol = "✓ SÍ" if stats['solution_found'] else "✗ NO"
            print(f"  {n:>5}×{n:<3}  | {ver_name:>20} | {stats['best_conflicts']:>12} | "
                  f"{stats['elapsed_time']:>10.2f} | {stats['total_generations']:>13} | {sol:>9}")
        print("-" * 80)

    print("\n  MEJORES SOLUCIONES (Cromosomas):")
    print("-" * 80)
    for n in sorted(results.keys()):
        for ver_key, ver_name in [('v1', 'V1'), ('v2', 'V2')]:
            best = results[n][ver_key]['best_individual']
            chrom_str = str(best.chromosome[:16])
            if len(best.chromosome) > 16:
                chrom_str = chrom_str[:-1] + ", ...]"
            print(f"  {n}×{n} {ver_name}: {chrom_str} | conflictos={best.conflicts}")
    print("=" * 80)

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   ALGORITMO GENÉTICO: PROBLEMA DE K-REINAS           ║")
    print("║   Versión 1: PMX + Mutación Swap                     ║")
    print("║   Versión 2: OX  + Mutación por Inversión            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"\nResultados se guardarán en: {OUTPUT_DIR}\n")

    # Ejecutar algoritmos
    t0 = time.time()
    results = run_all_boards(board_sizes=(8, 16, 24, 32), runs_per_board=3)
    total_time = time.time() - t0

    print(f"\nTiempo total de ejecución: {total_time:.1f}s")

    # Tabla en consola
    print_summary_table(results)

    # Gráficas
    fig_paths = generate_all_figures(results, OUTPUT_DIR)

    print(f"\n✓ ¡Todo completado! Archivos en: {OUTPUT_DIR}")
    return results


if __name__ == "__main__":
    main()