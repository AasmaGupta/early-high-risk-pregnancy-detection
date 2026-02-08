import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("edacolumn.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns)

df["State"] = df["State"].astype(str)



# Antenatal Care Coverage

plt.figure(figsize=(10,14))
sns.barplot(y="State", x="Antenatal_4_visits", data=df, palette="viridis")
plt.title("Antenatal Care Coverage (≥4 visits) across States", fontsize=14)
plt.xlabel("% Women with ≥4 ANC visits")
plt.ylabel("State")
plt.tight_layout()
plt.show()



# Women with Anaemia

plt.figure(figsize=(10,14))
sns.barplot(y="State", x="Anaemia_Women", data=df, palette="coolwarm")
plt.title("Anaemia in Women across States", fontsize=14)
plt.xlabel("% Women with Anaemia")
plt.ylabel("State")
plt.tight_layout()
plt.show()



# Place of Delivery (Institutional vs Home)

df_melted = df.melt(id_vars="State", 
                    value_vars=["Institutional_Delivery","Home_Delivery"],
                    var_name="DeliveryType", value_name="Percentage")

plt.figure(figsize=(10,14))
sns.barplot(y="State", x="Percentage", hue="DeliveryType", data=df_melted)
plt.title("Place of Delivery across States", fontsize=14)
plt.xlabel("% Deliveries")
plt.ylabel("State")
plt.legend(title="Delivery Type")
plt.tight_layout()
plt.show()



# Child Mortality (Infant, Neonatal, Under-5)

mortality_cols = ["Infant_Mortality", "Neonatal_Mortality", "Under5_Mortality"]

df[mortality_cols] = df[mortality_cols].apply(pd.to_numeric, errors="coerce")

plt.figure(figsize=(20,6))  
sns.heatmap(df[mortality_cols].T, annot=True, cmap="Reds", cbar=True, 
            xticklabels=df["State"], yticklabels=mortality_cols, fmt=".1f")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(fontsize=10)
plt.title("Child Mortality Indicators across States", fontsize=14)
plt.tight_layout()
plt.show()



# Female Education vs Maternal Care

plt.figure(figsize=(12,8))  # bigger for clarity
sns.scatterplot(x="Female_Literacy", y="Antenatal_4_visits", 
                data=df, s=120, color="teal", alpha=0.7)
plt.title("Female Literacy vs Antenatal Care Coverage", fontsize=14)
plt.xlabel("Female Literacy (%)")
plt.ylabel("ANC ≥4 visits (%)")
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



# Decision-Making Autonomy of Women

plt.figure(figsize=(10,14))
sns.barplot(y="State", x="Women_Autonomy", data=df, palette="magma")
plt.title("Women’s Autonomy in Healthcare Decisions across States", fontsize=14)
plt.xlabel("% Women with Autonomy")
plt.ylabel("State")
plt.tight_layout()
plt.show()



# Socio-Economic & Health Risk Link

plt.figure(figsize=(12,8))  
sns.scatterplot(x="Wealth_Index", y="Anaemia_Women", data=df, 
                size="Antenatal_4_visits", hue="Wealth_Index", 
                sizes=(50,300), palette="cool", alpha=0.7)
plt.title("Wealth vs Anaemia (Bubble size = ANC coverage)", fontsize=14)
plt.xlabel("Wealth Index (proxy)")
plt.ylabel("% Women Anaemic")
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()