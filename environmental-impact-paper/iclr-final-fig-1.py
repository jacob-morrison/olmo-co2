import matplotlib.pyplot as plt
import numpy as np

# Extract data for OLMo models (excluding the ones with missing values)
models = [
    "OLMo 20M", "OLMo 60M", "OLMo 150M", "OLMo 300M", "OLMo 700M", 
    "OLMo 7B", "OLMo 1B (3T)", "OLMo 7B (Twin)", "OLMo (04|07)24 7B",
    "OLMo 2 7B", "OLMo 2 13B", "OLMoE 0924"
]

# Carbon emissions (tCO2eq)
co2_emissions = [0.3, 0.4, 1, 2, 3, 22, 10, 70, 32, 52, 101, 18]

# Water consumption (kL)
water_consumption = [1, 1.6, 3.6, 5.9, 10, 87, 39, 487, 122, 202, 892, 70]

# Create the scatter plot
plt.figure(figsize=(12, 8))

# Set different marker sizes based on model size (just for visualization)
sizes = []
for model in models:
    if "20M" in model or "60M" in model or "150M" in model:
        sizes.append(100)
    elif "300M" in model or "700M" in model or "1B" in model:
        sizes.append(200)
    else:
        sizes.append(300)

# Create scatter plot with model names as annotations
scatter = plt.scatter(co2_emissions, water_consumption, s=sizes, alpha=0.7, c=range(len(models)), cmap='viridis')

# Annotate each point with model name
for i, model in enumerate(models):
    plt.annotate(model, (co2_emissions[i], water_consumption[i]),
                xytext=(7, 0), textcoords='offset points',
                fontsize=9, ha='left', va='center')

# Add log scales for better visualization since data spans multiple orders of magnitude
plt.xscale('log')
plt.yscale('log')

# Add labels and title
plt.xlabel('Carbon Emissions (tCO₂eq)', fontsize=12)
plt.ylabel('Water Consumption (kL)', fontsize=12)
plt.title('Carbon Emissions vs. Water Consumption for OLMo Models', fontsize=14)

# Add grid for better readability
plt.grid(True, alpha=0.3, which='both')

# Define color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# # Create reference lines for water and fuel usage
# # For water reference lines
# plt.axhline(y=113297, color=color_palette[3], linestyle='-', linewidth=2)
# plt.text(0.01, 95000, 'Yearly Water Usage (1 person, U.S.)', 
#          color=color_palette[3], va='center', ha='left', transform=plt.gca().get_yaxis_transform())

# plt.axhline(y=9442, color=color_palette[3], linestyle='-', linewidth=2)
# plt.text(0.01, 11000, 'Monthly Water Usage (1 person, U.S.)', 
#          color=color_palette[3], va='center', ha='left', transform=plt.gca().get_yaxis_transform())

# # For fuel reference lines (vertical lines)
# plt.axvline(x=0.8887, color=color_palette[2], linestyle='-', linewidth=2)
# plt.text(0.8887*1.2, min(water_consumption), '100 gallons of gas', 
#          color=color_palette[2], va='bottom', ha='left', rotation=90)

# plt.axvline(x=43, color=color_palette[2], linestyle='-', linewidth=2)
# plt.text(43*1.2, min(water_consumption), '100 barrels of oil', 
#          color=color_palette[2], va='bottom', ha='left', rotation=90)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()