# app/genetic_optimization.py
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
from .backtest import run_simple_backtest


def optimize_parameters_genetic():
    """Оптимизация параметров стратегии через генетический алгоритм"""

    # Определяем функцию приспособленности
    def evaluate(individual):
        # individual = [min_volume_factor, max_volatility, min_trend_strength, momentum_threshold]
        try:
            # Здесь нужно модифицировать стратегию с параметрами из individual
            # и запустить бэктест
            result = run_simple_backtest()  # Нужно адаптировать для принятия параметров

            # Целевая функция: максимизация Sharpe ratio и минимизация просадки
            total_return = result.metrics.get('total_return', 0)
            max_drawdown = result.metrics.get('max_drawdown', 1)
            win_rate = result.metrics.get('win_rate', 0)

            # Комбинированная метрика
            fitness = total_return * (1 - max_drawdown) * win_rate
            return fitness,

        except Exception as e:
            return -1,

    # Настройка генетического алгоритма
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Диапазоны параметров
    toolbox.register("attr_float", random.uniform, 0.8, 2.0)  # min_volume_factor
    toolbox.register("attr_float2", random.uniform, 0.02, 0.1)  # max_volatility
    toolbox.register("attr_float3", random.uniform, 0.05, 0.3)  # min_trend_strength
    toolbox.register("attr_float4", random.uniform, 0.001, 0.01)  # momentum_threshold

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_float, toolbox.attr_float2,
                      toolbox.attr_float3, toolbox.attr_float4), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Запуск оптимизации
    population = toolbox.population(n=20)
    generations = 10

    print("Starting genetic optimization...")
    for gen in range(generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))
        best = tools.selBest(population, k=1)[0]
        print(f"Generation {gen}: Best fitness = {best.fitness.values[0]:.4f}")

    best_params = tools.selBest(population, k=1)[0]
    print(f"\nBest parameters: {best_params}")

    return best_params