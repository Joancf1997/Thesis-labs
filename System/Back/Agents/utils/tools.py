def predict_population_linear(target_year):
    """Predicts the population for a given target year using a linear model.
    Args:
        target_year (int): The year for which to predict the population.
    Returns:
        int: The predicted population for the target year.
    """
    return int(target_year) * 2  

def calculator(a:int, b:int) -> int:
    """ Calculates the sum of two numbers.
    Args:
        a (int): The first number.
        b (int): The second number.
    Returns:
        int: The sum of the two numbers.
    """
    return int(a) + int(b)

def plot_population():
    """Plots the population colun in the dataset"""
    return "Plot generated!"

TASK_FUNCS = {
    "predict_population_linear": predict_population_linear,
    "calculator": calculator,
    "plot_population": plot_population
}