"""
Algoritmo Genético para el Problema de K-Reinas
================================================
Versión 1: Cruzamiento PMX (Partially Mapped Crossover) + Mutación por Intercambio
Versión 2: Cruzamiento OX (Order Crossover) + Mutación por Inversión

Autor: Algoritmo Genético K-Reinas
"""

import random
import time
import json
import os
import math
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict


# ============================================================
# REPRESENTACIÓN: Permutación (cromosoma = lista de posiciones)
# queens[i] = fila donde está la reina en la columna i
# Garantiza NO conflictos en filas ni columnas por diseño
# Solo necesitamos minimizar conflictos diagonales
# ============================================================

@dataclass
class GeneticConfig:
    """Configuración del algoritmo genético"""
    board_size: int = 8
    population_size: int = 200
    max_generations: int = 2000
    mutation_rate: float = 0.05
    crossover_rate: float = 0.90
    elitism_count: int = 5
    tournament_size: int = 5
    version: int = 1  # 1=PMX+Swap, 2=OX+Inversion
    max_no_improve: int = 300  # Parada temprana


@dataclass
class Individual:
    """Individuo del algoritmo genético"""
    chromosome: List[int]
    fitness: float = 0.0
    conflicts: int = 0

    def __post_init__(self):
        self.fitness, self.conflicts = calculate_fitness(self.chromosome)


def calculate_conflicts(chromosome: List[int]) -> int:
    """
    Calcula el número de conflictos diagonales.
    Como usamos permutación, no hay conflictos en filas ni columnas.
    """
    n = len(chromosome)
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(chromosome[i] - chromosome[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def calculate_fitness(chromosome: List[int]) -> Tuple[float, int]:
    """
    Fitness = 1 / (1 + conflicts)
    Fitness máximo = 1.0 cuando conflicts = 0
    """
    conflicts = calculate_conflicts(chromosome)
    fitness = 1.0 / (1.0 + conflicts)
    return fitness, conflicts


# ============================================================
# INICIALIZACIÓN
# ============================================================

def initialize_population(config: GeneticConfig) -> List[Individual]:
    """
    Genera población inicial con permutaciones aleatorias.
    Una permutación garantiza 1 reina por columna y 1 por fila.
    """
    population = []
    n = config.board_size

    # Agregar algunas soluciones "casi válidas" con heurística simple
    heuristic_count = max(5, config.population_size // 10)
    for _ in range(heuristic_count):
        chrom = list(range(n))
        # Heurística: intentar reducir conflictos diagonales inicialmente
        for _ in range(n * 2):
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            # Evaluar swap
            old_conf = calculate_conflicts(chrom)
            chrom[i], chrom[j] = chrom[j], chrom[i]
            new_conf = calculate_conflicts(chrom)
            if new_conf > old_conf:
                chrom[i], chrom[j] = chrom[j], chrom[i]  # revert
        population.append(Individual(chromosome=chrom[:]))

    # El resto es completamente aleatorio
    while len(population) < config.population_size:
        chrom = list(range(n))
        random.shuffle(chrom)
        population.append(Individual(chromosome=chrom))

    return population


# ============================================================
# SELECCIÓN: Torneo
# ============================================================

def tournament_selection(population: List[Individual], tournament_size: int) -> Individual:
    """Selección por torneo: elige el mejor de k individuos aleatorios"""
    candidates = random.sample(population, min(tournament_size, len(population)))
    return max(candidates, key=lambda ind: ind.fitness)


# ============================================================
# CRUZAMIENTO - VERSIÓN 1: PMX (Partially Mapped Crossover)
# ============================================================

def pmx_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Partially Mapped Crossover (PMX):
    - Selecciona un segmento de parent1 y lo copia al hijo
    - Los genes restantes se mapean desde parent2 evitando duplicados
    - Mantiene la propiedad de permutación
    """
    n = len(parent1)
    p1, p2 = sorted(random.sample(range(n), 2))

    child1 = [-1] * n
    child2 = [-1] * n

    # Copiar segmento seleccionado
    child1[p1:p2+1] = parent1[p1:p2+1]
    child2[p1:p2+1] = parent2[p1:p2+1]

    # Mapeo PMX para child1
    for i in range(p1, p2 + 1):
        val = parent2[i]
        if val not in child1[p1:p2+1]:
            # Encontrar posición
            pos = i
            while p1 <= pos <= p2:
                pos = parent2.index(parent1[pos])
            child1[pos] = val

    # Llenar restantes desde parent2
    for i in range(n):
        if child1[i] == -1:
            child1[i] = parent2[i]

    # Mapeo PMX para child2
    for i in range(p1, p2 + 1):
        val = parent1[i]
        if val not in child2[p1:p2+1]:
            pos = i
            while p1 <= pos <= p2:
                pos = parent1.index(parent2[pos])
            child2[pos] = val

    for i in range(n):
        if child2[i] == -1:
            child2[i] = parent1[i]

    return child1, child2


# ============================================================
# CRUZAMIENTO - VERSIÓN 2: OX (Order Crossover)
# ============================================================

def ox_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Order Crossover (OX):
    - Copia un segmento del padre 1 al hijo
    - Los elementos restantes se llenan en el orden en que aparecen en padre 2
    - Preserva el orden relativo de los genes
    """
    n = len(parent1)
    p1, p2 = sorted(random.sample(range(n), 2))

    def _ox(par1, par2):
        child = [-1] * n
        child[p1:p2+1] = par1[p1:p2+1]
        segment_set = set(par1[p1:p2+1])

        fill_values = [x for x in par2 if x not in segment_set]
        fill_idx = 0
        for i in list(range(p2 + 1, n)) + list(range(0, p1)):
            child[i] = fill_values[fill_idx]
            fill_idx += 1
        return child

    return _ox(parent1, parent2), _ox(parent2, parent1)


# ============================================================
# MUTACIÓN - VERSIÓN 1: Swap (Intercambio)
# ============================================================

def swap_mutation(chromosome: List[int], mutation_rate: float) -> List[int]:
    """
    Mutación por intercambio:
    - Selecciona dos posiciones aleatorias y las intercambia
    - Mantiene la propiedad de permutación
    """
    chrom = chromosome[:]
    n = len(chrom)
    for i in range(n):
        if random.random() < mutation_rate:
            j = random.randint(0, n - 1)
            chrom[i], chrom[j] = chrom[j], chrom[i]
    return chrom


# ============================================================
# MUTACIÓN - VERSIÓN 2: Inversión
# ============================================================

def inversion_mutation(chromosome: List[int], mutation_rate: float) -> List[int]:
    """
    Mutación por inversión:
    - Selecciona un subsegmento y lo invierte
    - Mantiene la propiedad de permutación
    - Preserva más estructura que swap
    """
    chrom = chromosome[:]
    if random.random() < mutation_rate:
        n = len(chrom)
        p1, p2 = sorted(random.sample(range(n), 2))
        chrom[p1:p2+1] = chrom[p1:p2+1][::-1]
    return chrom


# ============================================================
# REEMPLAZO: Elitismo + Generacional
# ============================================================

def elitism_replacement(old_pop: List[Individual], new_pop: List[Individual],
                         elitism_count: int) -> List[Individual]:
    """
    Reemplazo generacional con elitismo:
    - Los mejores 'elitism_count' individuos de la generación anterior sobreviven
    - El resto es reemplazado por la nueva generación
    """
    old_pop.sort(key=lambda x: x.fitness, reverse=True)
    elites = old_pop[:elitism_count]
    new_pop.sort(key=lambda x: x.fitness, reverse=True)
    combined = elites + new_pop[elitism_count:]
    return combined[:len(old_pop)]


# ============================================================
# ALGORITMO GENÉTICO PRINCIPAL
# ============================================================

class GeneticAlgorithm:
    def __init__(self, config: GeneticConfig):
        self.config = config
        self.version_name = "PMX + Swap Mutation" if config.version == 1 else "OX + Inversion Mutation"
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': [],
            'best_conflicts': [],
            'generations': [],
            'diversity': []
        }
        self.best_ever: Optional[Individual] = None
        self.solution_found = False
        self.solution_generation = -1

    def _crossover(self, p1, p2):
        if self.config.version == 1:
            return pmx_crossover(p1, p2)
        else:
            return ox_crossover(p1, p2)

    def _mutate(self, chrom):
        if self.config.version == 1:
            return swap_mutation(chrom, self.config.mutation_rate)
        else:
            return inversion_mutation(chrom, self.config.mutation_rate)

    def _diversity(self, population: List[Individual]) -> float:
        """Calcula diversidad como desviación estándar del fitness"""
        fitnesses = [ind.fitness for ind in population]
        return float(np.std(fitnesses))

    def run(self) -> Tuple[Individual, Dict]:
        """Ejecuta el algoritmo genético y retorna la mejor solución"""
        config = self.config
        start_time = time.time()

        # Inicialización
        population = initialize_population(config)
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_ever = deepcopy(population[0])

        no_improve = 0
        prev_best = self.best_ever.fitness

        for gen in range(config.max_generations):
            # Crear nueva generación
            new_population = []

            while len(new_population) < config.population_size:
                # Selección
                p1 = tournament_selection(population, config.tournament_size)
                p2 = tournament_selection(population, config.tournament_size)

                # Cruzamiento
                if random.random() < config.crossover_rate:
                    c1_chrom, c2_chrom = self._crossover(p1.chromosome, p2.chromosome)
                else:
                    c1_chrom, c2_chrom = p1.chromosome[:], p2.chromosome[:]

                # Mutación
                c1_chrom = self._mutate(c1_chrom)
                c2_chrom = self._mutate(c2_chrom)

                new_population.append(Individual(chromosome=c1_chrom))
                if len(new_population) < config.population_size:
                    new_population.append(Individual(chromosome=c2_chrom))

            # Reemplazo con elitismo
            population = elitism_replacement(population, new_population, config.elitism_count)
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Actualizar mejor
            if population[0].fitness > self.best_ever.fitness:
                self.best_ever = deepcopy(population[0])
                no_improve = 0
            else:
                no_improve += 1

            # Registro de estadísticas
            fitnesses = [ind.fitness for ind in population]
            self.history['generations'].append(gen)
            self.history['best_fitness'].append(population[0].fitness)
            self.history['avg_fitness'].append(float(np.mean(fitnesses)))
            self.history['worst_fitness'].append(min(fitnesses))
            self.history['best_conflicts'].append(population[0].conflicts)
            self.history['diversity'].append(self._diversity(population))

            # Solución encontrada
            if population[0].conflicts == 0:
                self.solution_found = True
                self.solution_generation = gen
                self.best_ever = deepcopy(population[0])
                break

            # Parada temprana por convergencia
            if no_improve >= config.max_no_improve:
                break

            # Reinicio parcial si hay convergencia prematura
            if self._diversity(population) < 0.001 and gen < config.max_generations * 0.8:
                # Reiniciar 30% de la población manteniendo élites
                n_reinit = int(config.population_size * 0.3)
                n = config.board_size
                for idx in range(config.elitism_count, config.elitism_count + n_reinit):
                    if idx < len(population):
                        chrom = list(range(n))
                        random.shuffle(chrom)
                        population[idx] = Individual(chromosome=chrom)

        elapsed = time.time() - start_time

        stats = {
            'version': self.config.version,
            'version_name': self.version_name,
            'board_size': config.board_size,
            'best_conflicts': self.best_ever.conflicts,
            'best_fitness': self.best_ever.fitness,
            'solution_found': self.solution_found,
            'solution_generation': self.solution_generation,
            'total_generations': len(self.history['generations']),
            'elapsed_time': round(elapsed, 3),
            'population_size': config.population_size,
            'mutation_rate': config.mutation_rate,
            'crossover_rate': config.crossover_rate,
        }

        return self.best_ever, stats


# ============================================================
# EJECUTOR MULTI-TABLERO
# ============================================================

def run_all_boards(board_sizes=(8, 16, 24, 32), runs_per_board=3):
    """Ejecuta ambas versiones del GA para todos los tamaños de tablero"""
    results = {}

    for n in board_sizes:
        print(f"\n{'='*60}")
        print(f"  Tablero {n}x{n}")
        print(f"{'='*60}")

        results[n] = {'v1': [], 'v2': []}

        # Configuración adaptativa según tamaño
        pop_size = min(500, max(100, n * 12))
        max_gen = min(5000, max(1000, n * 60))
        mut_rate = 0.08 if n <= 16 else 0.05

        for version in [1, 2]:
            ver_key = f'v{version}'
            ver_name = "PMX+Swap" if version == 1 else "OX+Inversión"
            print(f"\n  Versión {version} ({ver_name}):")

            best_overall = None
            best_stats = None
            all_histories = []

            for run in range(runs_per_board):
                config = GeneticConfig(
                    board_size=n,
                    population_size=pop_size,
                    max_generations=max_gen,
                    mutation_rate=mut_rate,
                    crossover_rate=0.90,
                    elitism_count=5,
                    tournament_size=5,
                    version=version,
                    max_no_improve=max_gen // 5
                )
                ga = GeneticAlgorithm(config)
                best, stats = ga.run()

                all_histories.append(ga.history)
                print(f"    Run {run+1}: conflictos={best.conflicts}, "
                      f"generaciones={stats['total_generations']}, "
                      f"tiempo={stats['elapsed_time']}s, "
                      f"solución={'✓' if stats['solution_found'] else '✗'}")

                if best_overall is None or best.conflicts < best_overall.conflicts:
                    best_overall = best
                    best_stats = stats

            results[n][ver_key] = {
                'best_individual': best_overall,
                'best_stats': best_stats,
                'all_histories': all_histories
            }

            print(f"    → Mejor: {best_overall.conflicts} conflictos")

    return results


if __name__ == "__main__":
    print("Ejecutando Algoritmos Genéticos para K-Reinas...")
    results = run_all_boards(board_sizes=(8, 16, 24, 32), runs_per_board=3)
    print("\n¡Completado!")