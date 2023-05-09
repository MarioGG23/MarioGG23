import os
import pathlib
ruta = pathlib.Path().absolute()
ruta = str(ruta) + '/trt_sh/'

try:
    os.mkdir(ruta)
except FileExistsError:
    print("Carpeta ya creada")
    

tipo = str(input('Tipo de transformacion: '))
modelo = str(input('Como se llama el modelo?: '))
ubicacion1 = '/home/rhernandez/onnx_models/' + modelo + '.onnx'

#ruta = '/Users/thadliguerra/Desktop/trt_sh/'
#ruta = '/home/rhernandez/trt_sh'

if tipo == 'fp32':
    trt =  ruta + 'trt_quantization_' + tipo + '_' + modelo + '.txt'
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
