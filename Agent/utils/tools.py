def predict_population_linear(target_year):
    return int(target_year) * 2  

def calculator(a, b):
    return int(a) + int(b)

def plot_population():
    return "Plot generated!"

TASK_FUNCS = {
    "predict_population_linear": predict_population_linear,
    "calculator": calculator,
    "plot_population": plot_population
}