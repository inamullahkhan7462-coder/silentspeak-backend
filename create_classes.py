import pickle

# Update this list to match the EXACT order of your 38 folders/labels
classes = [
    "Ain", "Aliph", "Bari yeh", "Bay", "Chay", "Chhoti yeh", "Daal", 
    "Daal_variant", "Dhaal", "Dhuaad", "Djay", "Fay", "Gaaf", "Ghain", 
    "Hamza", "Hay", "Jeem", "Kaaf", "Khay", "Laam", "Meem", "Noon", 
    "Pay", "Quaaf", "Ray", "Seen", "Sheen", "Suaad", "Tay", "Tey", 
    "Thay", "Toay'n", "Vao", "Zay", "Zoay'n", "aRay", "hey"
]

with open('classes.pkl', 'wb') as f:
    pickle.dump(classes, f)

print("Successfully created classes.pkl!")