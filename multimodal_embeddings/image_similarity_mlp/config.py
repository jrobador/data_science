color_channel = 3  
image_size = 32    
hidden_size = 256  
output_size = 64   
temperature = 0.07           
num_epochs = 100
log_interval = 10
embedding_dim = 10

# Mapping from number label to descriptive text
label_map = {
    0: "avión",
    1: "automóvil",
    2: "pájaro",
    3: "gato",
    4: "ciervo",
    5: "perro",
    6: "rana",
    7: "caballo",
    8: "barco",
    9: "camión"
}
#Join all dict in one array. Split them in words (tokenize). Take only unique words and calculate its length
vocab_size = len(set(" ".join(label_map.values()).split()))


print (set(" ".join(label_map.values()).split()))
