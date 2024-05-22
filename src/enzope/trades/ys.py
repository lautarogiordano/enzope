def yard_sale(ri, wi, rj, wj):
    """
    Calculate the Yard-Sale interaction between two agents.

    Parameters:
    ri (int): Risk value of the first agent.
    wi (int): Wealth value of the first agent.
    rj (int): Risk value of the second agent.
    wj (int): Wealth value of the second agent.

    Returns:
    int: The minimum value between ri * wi and rj * wj.
    """
    return min(ri * wi, rj * wj)
