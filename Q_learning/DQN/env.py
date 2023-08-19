class SimpleEnvironment:
    def __init__(self, initial_price=1.0):
        self.current_price = initial_price

    def step(self, action):
        # Increase, keep the same, or decrease the price based on the action
        if action == 0:
            self.current_price *= 1.1
        elif action == 2:
            self.current_price *= 0.9

        # Here, reward is the new price (this is a simple and naive reward function for demonstration)
        reward = self.current_price

        # Next state is the new price
        next_state = [self.current_price]

        # Simple condition to check if the episode is done (for demonstration purposes)
        done = True if self.current_price > 2.0 or self.current_price < 0.5 else False

        return next_state, reward, done

    def reset(self):
        self.current_price = 1.0
        return [self.current_price]
