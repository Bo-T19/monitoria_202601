def calcular_promedio(nombre, notas, bono=0.1):

    promedio = sum(notas)/len(notas) 
    print ('El promedio de ' + nombre + " es: "+ str(promedio)+', con el bono es: ' +str(promedio + bono))
    return promedio + bono

def saludar(nombre):
    """
    Función para imprimir un saludo
    """
    print("Hola, " + nombre)