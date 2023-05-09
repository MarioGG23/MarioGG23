import os
print('Onnx a trt')
tipo = str(input('Tipo de transformacion: '))
#ubicacion1 = modelo_convert #ruta del archivo.onnx con el archivo incluido
modelo = str(input('Como se llama el modelo?: '))
ubicacion1 = '/home/rhernandez/onnx_models/' + modelo + '.onnx'
ubicacion1_sinext = ubicacion1.split('.')

#trt = aux2 + 'trt_quantization.txt' #aux 2 es donde se va a guardar el trt
#ruta = '/Users/thadliguerra/Desktop/trt_sh/'
ruta = '/home/rhernandez/trt_sh/'

if tipo == 'fp32':
    trt =  ruta + 'trt_quantization_' + tipo + '_' + modelo + '.txt'
    print(tipo)
    #ubicacion2 = ubicacion1_sinext[0] + '_batch1_fp32.trt' 
    ubicacion2 = '/home/rhernandez/modelos_trt/' + modelo + '_batch1_fp32.trt'
    linea1 = '/usr/src/tensorrt/bin/trtexec --onnx=' + ubicacion1 + ' --saveEngine=' + ubicacion2
    linea2 = '/usr/src/tensorrt/bin/trtexec --loadEngine=' + ubicacion2
    file = open(trt, "w")
    file.write(linea1 + os.linesep)
    file.write(linea2)
    file.close()
    trtaux = trt.split('.')
    trt_sh = trtaux[0] + '.sh'
    os.rename(trt, trt_sh)
    
elif tipo == 'fp16':
    trt =  ruta + 'trt_quantization_' + tipo + '_' + modelo + '.txt'
    print(tipo)
    #ubicacion2 = ubicacion1_sinext[0] + '_batch1_fp16.trt' 
    ubicacion2 = '/home/rhernandez/modelos_trt/' + modelo + '_batch1_fp16.trt'
    linea1 = '/usr/src/tensorrt/bin/trtexec --onnx=' + ubicacion1 + ' --saveEngine=' + ubicacion2
    linea2 = '/usr/src/tensorrt/bin/trtexec --loadEngine=' + ubicacion2
    file = open(trt, "w")
    file.write(linea1 + os.linesep)
    file.write(linea2)
    file.close()
    trtaux = trt.split('.')
    trt_sh = trtaux[0] + '.sh'
    os.rename(trt, trt_sh)
    
else:
    trt =  ruta + 'trt_quantization_' + tipo + '_' + modelo + '.txt'
    print(tipo)
    #ubicacion2 = ubicacion1_sinext[0] + '_batch1_int8.trt'
    ubicacion2 = '/home/rhernandez/modelos_trt/' + modelo + '_batch1_int8.trt'
    linea1 = '/usr/src/tensorrt/bin/trtexec --onnx=' + ubicacion1 + ' --saveEngine=' + ubicacion2
    linea2 = '/usr/src/tensorrt/bin/trtexec --loadEngine=' + ubicacion2
    file = open(trt, "w")
    file.write(linea1 + os.linesep)
    file.write(linea2)
    file.close()
    trtaux = trt.split('.')
    trt_sh = trtaux[0] + '.sh'
    os.rename(trt, trt_sh)
