"""
Visualización y Gráficas para K-Reinas con Algoritmo Genético
=============================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import json
import os
from typing import Dict, List, Optional


# ============================================================
# PALETA DE COLORES
# ============================================================
COLORS = {
    'bg_dark': '#0a0e1a',
    'bg_card': '#111827',
    'accent_blue': '#3b82f6',
    'accent_purple': '#8b5cf6',
    'accent_green': '#10b981',
    'accent_orange': '#f59e0b',
    'accent_red': '#ef4444',
    'text_light': '#f8fafc',
    'text_muted': '#94a3b8',
    'grid_line': '#1e293b',
    'queen_v1': '#fbbf24',
    'queen_v2': '#a78bfa',
    'conflict': '#ef4444',
    'safe': '#10b981',
    'board_light': '#e2e8f0',
    'board_dark': '#475569',
}


# ============================================================
# TABLERO DE AJEDREZ
# ============================================================

def draw_chessboard(ax, chromosome: List[int], title: str, version: int = 1):
    """
    Dibuja el tablero de ajedrez con las reinas posicionadas.
    Verde = sin conflictos, Rojo = conflicto diagonal
    """
    n = len(chromosome)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')

    queen_color = COLORS['queen_v1'] if version == 1 else COLORS['queen_v2']

    # Dibujar casillas
    for row in range(n):
        for col in range(n):
            is_light = (row + col) % 2 == 0
            color = '#c8d6e5' if is_light else '#2d3748'
            rect = patches.Rectangle((col, n - 1 - row), 1, 1,
                                       linewidth=0, facecolor=color, alpha=0.9)
            ax.add_patch(rect)

    # Detectar reinas con conflictos
    conflicts_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if abs(chromosome[i] - chromosome[j]) == abs(i - j):
                conflicts_set.add(i)
                conflicts_set.add(j)

    # Dibujar reinas
    for col, row in enumerate(chromosome):
        y = n - 1 - row
        x = col

        # Sombra
        shadow = plt.Circle((x + 0.55, y + 0.45), 0.28, color='#000000', alpha=0.25)
        ax.add_patch(shadow)

        # Color según conflicto
        if col in conflicts_set:
            q_color = COLORS['conflict']
            ring_color = '#ff6b6b'
        else:
            q_color = queen_color
            ring_color = '#ffffff'

        # Círculo de la reina
        circle_outer = plt.Circle((x + 0.5, y + 0.5), 0.32, color=ring_color, alpha=0.3)
        circle = plt.Circle((x + 0.5, y + 0.5), 0.26, color=q_color, zorder=5)
        ax.add_patch(circle_outer)
        ax.add_patch(circle)

        # Símbolo ♛
        ax.text(x + 0.5, y + 0.5, '♛', ha='center', va='center',
                fontsize=max(6, min(14, 120 // n)), color='white',
                fontweight='bold', zorder=6)

    # Borde del tablero
    border = patches.Rectangle((0, 0), n, n, linewidth=2,
                                 edgecolor=queen_color, facecolor='none', alpha=0.8)
    ax.add_patch(border)

    # Etiquetas de columnas (si el tablero no es muy grande)
    if n <= 16:
        for i in range(n):
            ax.text(i + 0.5, -0.35, str(i + 1), ha='center', va='center',
                    fontsize=max(5, 8 - n // 4), color=COLORS['text_muted'])
            ax.text(-0.35, n - i - 0.5, str(i + 1), ha='center', va='center',
                    fontsize=max(5, 8 - n // 4), color=COLORS['text_muted'])

    ax.set_title(title, color=COLORS['text_light'], fontsize=10, pad=8, fontweight='bold')


# ============================================================
# GRÁFICAS DE CONVERGENCIA
# ============================================================

def plot_convergence(ax, history: Dict, title: str, color: str):
    """Gráfica de convergencia del fitness"""
    gens = history['generations']
    best = history['best_fitness']
    avg = history['avg_fitness']
    worst = history['worst_fitness']

    ax.set_facecolor(COLORS['bg_card'])
    ax.fill_between(gens, worst, best, alpha=0.15, color=color)
    ax.plot(gens, worst, color=color, alpha=0.3, linewidth=0.8, linestyle='--')
    ax.plot(gens, avg, color=color, alpha=0.6, linewidth=1.2, linestyle='-', label='Promedio')
    ax.plot(gens, best, color=color, linewidth=2.0, label='Mejor')

    ax.set_xlabel('Generación', color=COLORS['text_muted'], fontsize=8)
    ax.set_ylabel('Fitness', color=COLORS['text_muted'], fontsize=8)
    ax.set_title(title, color=COLORS['text_light'], fontsize=9, fontweight='bold')
    ax.tick_params(colors=COLORS['text_muted'], labelsize=7)
    ax.spines['bottom'].set_color(COLORS['grid_line'])
    ax.spines['left'].set_color(COLORS['grid_line'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color=COLORS['grid_line'])
    ax.legend(fontsize=7, labelcolor=COLORS['text_muted'],
              facecolor=COLORS['bg_dark'], edgecolor=COLORS['grid_line'])


def plot_conflicts_over_time(ax, history: Dict, title: str, color: str):
    """Gráfica de conflictos a lo largo de generaciones"""
    gens = history['generations']
    conflicts = history['best_conflicts']

    ax.set_facecolor(COLORS['bg_card'])
    ax.fill_between(gens, conflicts, alpha=0.2, color=color)
    ax.plot(gens, conflicts, color=color, linewidth=2.0)
    ax.axhline(y=0, color=COLORS['accent_green'], linewidth=1.5,
               linestyle='--', alpha=0.7, label='Solución óptima (0)')

    ax.set_xlabel('Generación', color=COLORS['text_muted'], fontsize=8)
    ax.set_ylabel('Conflictos', color=COLORS['text_muted'], fontsize=8)
    ax.set_title(title, color=COLORS['text_light'], fontsize=9, fontweight='bold')
    ax.tick_params(colors=COLORS['text_muted'], labelsize=7)
    ax.spines['bottom'].set_color(COLORS['grid_line'])
    ax.spines['left'].set_color(COLORS['grid_line'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color=COLORS['grid_line'])
    ax.legend(fontsize=7, labelcolor=COLORS['text_muted'],
              facecolor=COLORS['bg_dark'], edgecolor=COLORS['grid_line'])


def plot_diversity(ax, history: Dict, title: str, color: str):
    """Gráfica de diversidad de la población"""
    gens = history['generations']
    diversity = history['diversity']

    ax.set_facecolor(COLORS['bg_card'])
    ax.fill_between(gens, diversity, alpha=0.25, color=color)
    ax.plot(gens, diversity, color=color, linewidth=1.5)

    ax.set_xlabel('Generación', color=COLORS['text_muted'], fontsize=8)
    ax.set_ylabel('Diversidad (σ fitness)', color=COLORS['text_muted'], fontsize=8)
    ax.set_title(title, color=COLORS['text_light'], fontsize=9, fontweight='bold')
    ax.tick_params(colors=COLORS['text_muted'], labelsize=7)
    ax.spines['bottom'].set_color(COLORS['grid_line'])
    ax.spines['left'].set_color(COLORS['grid_line'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color=COLORS['grid_line'])


def plot_comparison_bar(ax, results_summary: Dict):
    """Gráfica de barras comparando conflictos por versión y tamaño"""
    board_sizes = sorted(results_summary.keys())
    x = np.arange(len(board_sizes))
    width = 0.35

    v1_conflicts = [results_summary[n]['v1'] for n in board_sizes]
    v2_conflicts = [results_summary[n]['v2'] for n in board_sizes]

    bars1 = ax.bar(x - width/2, v1_conflicts, width, label='V1: PMX + Swap',
                    color=COLORS['accent_blue'], alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, v2_conflicts, width, label='V2: OX + Inversión',
                    color=COLORS['accent_purple'], alpha=0.85, edgecolor='white', linewidth=0.5)

    # Etiquetas en barras
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                ha='center', va='bottom', color=COLORS['text_light'], fontsize=8, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                ha='center', va='bottom', color=COLORS['text_light'], fontsize=8, fontweight='bold')

    ax.set_facecolor(COLORS['bg_card'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}×{n}' for n in board_sizes], color=COLORS['text_muted'])
    ax.set_ylabel('Conflictos (mejor solución)', color=COLORS['text_muted'], fontsize=9)
    ax.set_title('Comparativa: Conflictos por Versión y Tamaño', color=COLORS['text_light'],
                  fontsize=10, fontweight='bold')
    ax.tick_params(colors=COLORS['text_muted'], labelsize=8)
    ax.spines['bottom'].set_color(COLORS['grid_line'])
    ax.spines['left'].set_color(COLORS['grid_line'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, axis='y', color=COLORS['grid_line'])
    ax.legend(fontsize=8, labelcolor=COLORS['text_muted'],
              facecolor=COLORS['bg_dark'], edgecolor=COLORS['grid_line'])
    ax.axhline(y=0, color=COLORS['accent_green'], linewidth=1.5, linestyle='--',
               alpha=0.6, label='Óptimo')


def plot_time_comparison(ax, time_summary: Dict):
    """Gráfica de tiempo de ejecución"""
    board_sizes = sorted(time_summary.keys())
    x = np.arange(len(board_sizes))
    width = 0.35

    v1_times = [time_summary[n]['v1'] for n in board_sizes]
    v2_times = [time_summary[n]['v2'] for n in board_sizes]

    ax.plot(board_sizes, v1_times, 'o-', color=COLORS['accent_blue'],
            linewidth=2, markersize=7, label='V1: PMX + Swap')
    ax.plot(board_sizes, v2_times, 's-', color=COLORS['accent_purple'],
            linewidth=2, markersize=7, label='V2: OX + Inversión')
    ax.fill_between(board_sizes, v1_times, alpha=0.15, color=COLORS['accent_blue'])
    ax.fill_between(board_sizes, v2_times, alpha=0.15, color=COLORS['accent_purple'])

    ax.set_facecolor(COLORS['bg_card'])
    ax.set_xticks(board_sizes)
    ax.set_xticklabels([f'{n}×{n}' for n in board_sizes], color=COLORS['text_muted'])
    ax.set_ylabel('Tiempo (segundos)', color=COLORS['text_muted'], fontsize=9)
    ax.set_title('Tiempo de Ejecución por Versión', color=COLORS['text_light'],
                  fontsize=10, fontweight='bold')
    ax.tick_params(colors=COLORS['text_muted'], labelsize=8)
    ax.spines['bottom'].set_color(COLORS['grid_line'])
    ax.spines['left'].set_color(COLORS['grid_line'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2, color=COLORS['grid_line'])
    ax.legend(fontsize=8, labelcolor=COLORS['text_muted'],
              facecolor=COLORS['bg_dark'], edgecolor=COLORS['grid_line'])


# ============================================================
# FIGURA PRINCIPAL: Tableros con soluciones
# ============================================================

def generate_boards_figure(results: Dict, output_dir: str):
    """Genera figura con los tableros de las mejores soluciones"""
    board_sizes = sorted(results.keys())
    n_boards = len(board_sizes)

    fig = plt.figure(figsize=(20, 5 * n_boards), facecolor=COLORS['bg_dark'])
    fig.suptitle('K-Reinas: Mejores Soluciones Encontradas',
                  color=COLORS['text_light'], fontsize=16, fontweight='bold', y=0.98)

    for row_idx, n in enumerate(board_sizes):
        for ver_idx, ver_key in enumerate(['v1', 'v2']):
            ver_num = 1 if ver_key == 'v1' else 2
            ax = fig.add_subplot(n_boards, 2, row_idx * 2 + ver_idx + 1,
                                  facecolor=COLORS['bg_card'])

            data = results[n][ver_key]
            best = data['best_individual']
            stats = data['best_stats']

            ver_name = "V1: PMX + Swap" if ver_num == 1 else "V2: OX + Inversión"
            sol_str = "✓ SOLUCIÓN EXACTA" if stats['solution_found'] else f"✗ {best.conflicts} conflictos"
            title = f"{n}×{n} | {ver_name}\n{sol_str}"

            draw_chessboard(ax, best.chromosome, title, ver_num)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(output_dir, 'tableros_soluciones.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close()
    print(f"  → Guardado: {path}")
    return path

# ============================================================
# FIGURA: Comparativa global
# ============================================================

def generate_comparison_figure(results: Dict, output_dir: str):
    """Genera figura comparativa entre ambas versiones"""
    board_sizes = sorted(results.keys())

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), facecolor=COLORS['bg_dark'])
    fig.suptitle('Análisis Comparativo: V1 (PMX+Swap) vs V2 (OX+Inversión)',
                  color=COLORS['text_light'], fontsize=14, fontweight='bold')

    # Conflictos por versión
    conflict_summary = {n: {
        'v1': results[n]['v1']['best_stats']['best_conflicts'],
        'v2': results[n]['v2']['best_stats']['best_conflicts']
    } for n in board_sizes}
    plot_comparison_bar(axes[0, 0], conflict_summary)

    # Tiempos
    time_summary = {n: {
        'v1': results[n]['v1']['best_stats']['elapsed_time'],
        'v2': results[n]['v2']['best_stats']['elapsed_time']
    } for n in board_sizes}
    plot_time_comparison(axes[0, 1], time_summary)

    # Generaciones hasta solución
    ax = axes[0, 2]
    ax.set_facecolor(COLORS['bg_card'])
    v1_gens = [results[n]['v1']['best_stats']['total_generations'] for n in board_sizes]
    v2_gens = [results[n]['v2']['best_stats']['total_generations'] for n in board_sizes]
    ax.plot(board_sizes, v1_gens, 'o-', color=COLORS['accent_blue'],
            linewidth=2, markersize=8, label='V1: PMX+Swap')
    ax.plot(board_sizes, v2_gens, 's-', color=COLORS['accent_purple'],
            linewidth=2, markersize=8, label='V2: OX+Inversión')
    ax.fill_between(board_sizes, v1_gens, alpha=0.15, color=COLORS['accent_blue'])
    ax.fill_between(board_sizes, v2_gens, alpha=0.15, color=COLORS['accent_purple'])
    ax.set_title('Generaciones Ejecutadas', color=COLORS['text_light'], fontweight='bold')
    ax.set_ylabel('Generaciones', color=COLORS['text_muted'])
    ax.set_xticks(board_sizes)
    ax.set_xticklabels([f'{n}×{n}' for n in board_sizes], color=COLORS['text_muted'])
    ax.tick_params(colors=COLORS['text_muted'])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid_line']); ax.spines['left'].set_color(COLORS['grid_line'])
    ax.grid(True, alpha=0.2, color=COLORS['grid_line'])
    ax.legend(fontsize=8, labelcolor=COLORS['text_muted'],
              facecolor=COLORS['bg_dark'], edgecolor=COLORS['grid_line'])

    # Diversidad promedio V1
    ax = axes[1, 0]
    for n, color in zip(board_sizes, [COLORS['accent_blue'], COLORS['accent_green'],
                                        COLORS['accent_orange'], COLORS['accent_red']]):
        hist = results[n]['v1']['all_histories'][0]
        ax.plot(hist['generations'], hist['diversity'],
                linewidth=1.5, label=f'{n}×{n}', color=color)
    ax.set_facecolor(COLORS['bg_card'])
    ax.set_title('Diversidad V1 (PMX+Swap)', color=COLORS['text_light'], fontweight='bold')
    ax.set_xlabel('Generación', color=COLORS['text_muted'])
    ax.set_ylabel('σ Fitness', color=COLORS['text_muted'])
    ax.tick_params(colors=COLORS['text_muted'])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid_line']); ax.spines['left'].set_color(COLORS['grid_line'])
    ax.grid(True, alpha=0.2, color=COLORS['grid_line'])
    ax.legend(fontsize=8, labelcolor=COLORS['text_muted'],
              facecolor=COLORS['bg_dark'], edgecolor=COLORS['grid_line'])

    # Diversidad promedio V2
    ax = axes[1, 1]
    for n, color in zip(board_sizes, [COLORS['accent_blue'], COLORS['accent_green'],
                                        COLORS['accent_orange'], COLORS['accent_red']]):
        hist = results[n]['v2']['all_histories'][0]
        ax.plot(hist['generations'], hist['diversity'],
                linewidth=1.5, label=f'{n}×{n}', color=color)
    ax.set_facecolor(COLORS['bg_card'])
    ax.set_title('Diversidad V2 (OX+Inversión)', color=COLORS['text_light'], fontweight='bold')
    ax.set_xlabel('Generación', color=COLORS['text_muted'])
    ax.set_ylabel('σ Fitness', color=COLORS['text_muted'])
    ax.tick_params(colors=COLORS['text_muted'])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color(COLORS['grid_line']); ax.spines['left'].set_color(COLORS['grid_line'])
    ax.grid(True, alpha=0.2, color=COLORS['grid_line'])
    ax.legend(fontsize=8, labelcolor=COLORS['text_muted'],
              facecolor=COLORS['bg_dark'], edgecolor=COLORS['grid_line'])

    # Tabla de resumen
    ax = axes[1, 2]
    ax.set_facecolor(COLORS['bg_card'])
    ax.axis('off')
    table_data = []
    headers = ['Tablero', 'V1 Conf.', 'V1 Tiempo', 'V2 Conf.', 'V2 Tiempo']
    for n in board_sizes:
        s1 = results[n]['v1']['best_stats']
        s2 = results[n]['v2']['best_stats']
        sol1 = '✓' if s1['solution_found'] else f"{s1['best_conflicts']}"
        sol2 = '✓' if s2['solution_found'] else f"{s2['best_conflicts']}"
        table_data.append([f'{n}×{n}', sol1, f"{s1['elapsed_time']}s",
                            sol2, f"{s2['elapsed_time']}s"])

    table = ax.table(cellText=table_data, colLabels=headers,
                      cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#1e3a5f')
            cell.set_text_props(color=COLORS['text_light'], fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#1a2035')
            cell.set_text_props(color=COLORS['text_light'])
        else:
            cell.set_facecolor('#111827')
            cell.set_text_props(color=COLORS['text_light'])
        cell.set_edgecolor(COLORS['grid_line'])

    ax.set_title('Resumen de Resultados', color=COLORS['text_light'],
                  fontweight='bold', pad=15)

    for ax in axes.flat:
        ax.set_facecolor(COLORS['bg_card'])

    plt.tight_layout()
    path = os.path.join(output_dir, 'comparativa.png')
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor=COLORS['bg_dark'])
    plt.close()
    print(f"  → Guardado: {path}")
    return path


def generate_all_figures(results: Dict, output_dir: str):
    """Genera todas las figuras"""
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerando gráficas...")
    paths = []
    paths.append(generate_boards_figure(results, output_dir))
    paths.append(generate_comparison_figure(results, output_dir))
    return paths