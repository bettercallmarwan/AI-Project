import tkinter as tk
from tkinter import ttk, messagebox
import random
import matplotlib.pyplot as plt
import statistics

class Individual:
    def __init__(self, size, max_weight, weights, values, is_01_knapsack=True):
        self.size = size
        self.max_weight = max_weight
        self.weights = weights
        self.values = values
        self.is_01_knapsack = is_01_knapsack
        self.chromosome = self.generate_chromosome()
        self.fitness = self.calculate_fitness()

    def generate_chromosome(self):
        if self.is_01_knapsack:
            return [random.randint(0, 1) for _ in range(self.size)]
        else:
            max_items = [self.max_weight // w if w > 0 else 0 for w in self.weights]
            return [random.randint(0, max_items[i]) for i in range(self.size)]

    def calculate_fitness(self):
        total_weight = sum(self.chromosome[i] * self.weights[i] for i in range(self.size))
        if total_weight > self.max_weight:
            return 0
        return sum(self.chromosome[i] * self.values[i] for i in range(self.size))


class GeneticAlgorithm:
    def __init__(self, population_size, size, max_weight, weights, values, is_01_knapsack=True):
        self.population_size = population_size
        self.size = size
        self.max_weight = max_weight
        self.weights = weights
        self.values = values
        self.is_01_knapsack = is_01_knapsack
        self.population = []
        self.best_fitness_history = []  # Track best fitness over generations
        self.initialize_population()

    def initialize_population(self):
        self.population = []
        while len(self.population) < self.population_size:
            individual = Individual(self.size, self.max_weight, self.weights, self.values, self.is_01_knapsack)
            if individual.fitness > 0:  # Only add valid solutions
                self.population.append(individual)

    def select_parent(self):
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        return random.choice(tournament[:2])

    def crossover(self, parent1, parent2):
        if random.random() < 0.8:  # 80% crossover rate
            crossover_point = random.randint(1, self.size - 1)
            child1_chromosome = parent1.chromosome[:crossover_point] + parent2.chromosome[crossover_point:]
            child2_chromosome = parent2.chromosome[:crossover_point] + parent1.chromosome[crossover_point:]
        else:
            child1_chromosome = parent1.chromosome[:]
            child2_chromosome = parent2.chromosome[:]
        return child1_chromosome, child2_chromosome

    def mutate(self, chromosome):
        mutation_rate = 0.1  # Fixed mutation rate
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                if self.is_01_knapsack:
                    chromosome[i] = 1 - chromosome[i]
                else:
                    max_items = self.max_weight // self.weights[i] if self.weights[i] > 0 else 0
                    chromosome[i] = random.randint(0, max_items)
        return chromosome

    def evolve(self):
        # Elitism: Keep the best solutions
        elite_size = max(2, self.population_size // 20)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population = self.population[:elite_size]

        # Generate new solutions
        while len(new_population) < self.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()

            child1_chromosome, child2_chromosome = self.crossover(parent1, parent2)

            child1_chromosome = self.mutate(child1_chromosome)
            child2_chromosome = self.mutate(child2_chromosome)

            for chromosome in [child1_chromosome, child2_chromosome]:
                child = Individual(self.size, self.max_weight, self.weights, 
                                self.values, self.is_01_knapsack)
                child.chromosome = chromosome
                child.fitness = child.calculate_fitness()
                if child.fitness > 0:  # Only add valid solutions
                    new_population.append(child)
                if len(new_population) >= self.population_size:
                    break

        self.population = new_population[:self.population_size]
        
        # Track best fitness in this generation
        best_solution = max(self.population, key=lambda x: x.fitness)
        self.best_fitness_history.append(best_solution.fitness)
        
        return best_solution
# The rest of the code (KnapsackSolverGUI class and main function) remains the same

class KnapsackSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Knapsack Problem Solver")
        self.create_widgets()

    def create_widgets(self):
        self.input_frame = ttk.LabelFrame(self.root, text="Input Parameters", padding="10")
        self.input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        self.items_outer_frame = ttk.LabelFrame(self.root, text="Items", padding="10")
        self.items_outer_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        self.canvas = tk.Canvas(self.items_outer_frame)
        self.scrollbar = ttk.Scrollbar(self.items_outer_frame, orient="vertical", command=self.canvas.yview)
        self.items_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas_frame = self.canvas.create_window((0, 0), window=self.items_frame, anchor="nw")

        self.items_frame.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        self.output_frame = ttk.LabelFrame(self.root, text="Results", padding="10")
        self.output_frame.grid(row=0, column=1, rowspan=2, padx=10, pady=5, sticky="nsew")

        ttk.Label(self.input_frame, text="Problem Type:").grid(row=0, column=0, padx=5, pady=5)
        self.problem_type = tk.StringVar(value="0/1 Knapsack")
        ttk.Radiobutton(self.input_frame, text="0/1 Knapsack", variable=self.problem_type, value="0/1 Knapsack").grid(row=0, column=1)
        ttk.Radiobutton(self.input_frame, text="Unbounded Knapsack", variable=self.problem_type, value="Unbounded Knapsack").grid(row=0, column=2)

        ttk.Label(self.input_frame, text="Number of Items:").grid(row=1, column=0, padx=5, pady=5)
        self.n_items = tk.StringVar(value="5")
        ttk.Entry(self.input_frame, textvariable=self.n_items).grid(row=1, column=1)

        ttk.Label(self.input_frame, text="Maximum Weight:").grid(row=2, column=0, padx=5, pady=5)
        self.max_weight = tk.StringVar(value="100")
        ttk.Entry(self.input_frame, textvariable=self.max_weight).grid(row=2, column=1)

        ttk.Label(self.input_frame, text="Number of Generations:").grid(row=3, column=0, padx=5, pady=5)
        self.n_generations = tk.StringVar(value="100")
        ttk.Entry(self.input_frame, textvariable=self.n_generations).grid(row=3, column=1)

        ttk.Button(self.input_frame, text="Generate Item Fields", command=self.generate_item_fields).grid(row=4, column=0, columnspan=2, pady=10)

        self.output_scrollbar = ttk.Scrollbar(self.output_frame)
        self.output_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_text = tk.Text(self.output_frame, width=50, height=30, yscrollcommand=self.output_scrollbar.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.output_scrollbar.config(command=self.output_text.yview)

    def on_frame_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event):
        width = event.width
        self.canvas.itemconfig(self.canvas_frame, width=width)

    def on_mousewheel(self, event):
        if self.items_outer_frame.winfo_height() < self.items_frame.winfo_height():
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def generate_item_fields(self):
        try:
            n = int(self.n_items.get())
            for widget in self.items_frame.winfo_children():
                widget.destroy()

            ttk.Label(self.items_frame, text="Item").grid(row=0, column=0, padx=5, pady=5)
            ttk.Label(self.items_frame, text="Weight").grid(row=0, column=1, padx=5, pady=5)
            ttk.Label(self.items_frame, text="Value").grid(row=0, column=2, padx=5, pady=5)

            self.weight_entries = []
            self.value_entries = []
            for i in range(n):
                ttk.Label(self.items_frame, text=f"Item {i+1}").grid(row=i+1, column=0, padx=5, pady=2)

                weight_var = tk.StringVar(value=str(random.randint(1, 30)))
                weight_entry = ttk.Entry(self.items_frame, textvariable=weight_var)
                weight_entry.grid(row=i+1, column=1, padx=5, pady=2)
                self.weight_entries.append(weight_var)

                value_var = tk.StringVar(value=str(random.randint(10, 100)))
                value_entry = ttk.Entry(self.items_frame, textvariable=value_var)
                value_entry.grid(row=i+1, column=2, padx=5, pady=2)
                self.value_entries.append(value_var)

            ttk.Button(self.items_frame, text="Solve", command=self.solve_knapsack).grid(row=n+1, column=0, columnspan=3, pady=10)

        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of items")

    def solve_knapsack(self):
        try:
            size = int(self.n_items.get())
            max_weight = int(self.max_weight.get())
            n_generations = int(self.n_generations.get())
            weights = [int(weight.get()) for weight in self.weight_entries]
            values = [int(value.get()) for value in self.value_entries]
            is_01_knapsack = self.problem_type.get() == "0/1 Knapsack"

            self.output_text.delete(1.0, tk.END)

            ga = GeneticAlgorithm(100, size, max_weight, weights, values, is_01_knapsack)

            for gen in range(n_generations):
                best_solution = ga.evolve()

                total_weight = sum(best_solution.chromosome[i] * weights[i] for i in range(size))

                result = f"Generation {gen + 1}:\n"
                result += f"Total Value: {best_solution.fitness}\n"
                result += f"Total Weight: {total_weight}/{max_weight}\n"

                result += "Selected Items: "
                selected = []
                for i in range(size):
                    if best_solution.chromosome[i] > 0:
                        selected.append(f"Item {i+1}({best_solution.chromosome[i]} units)")

                result += ", ".join(selected) + "\n"
                result += "-" * 50 + "\n"

                self.output_text.insert(tk.END, result)
                self.output_text.see(tk.END)
                self.root.update()

            # Plot best fitness over time
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, n_generations + 1), ga.best_fitness_history, 'b-')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness Value')
            plt.title('Best Solution Quality Over Generations')
            plt.grid(True)
            plt.show()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all fields")

def main():
    root = tk.Tk()
    app = KnapsackSolverGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()