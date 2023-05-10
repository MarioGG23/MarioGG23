import os
import pathlib
#ruta = pathlib.Path().absolute()
#ruta = str(ruta) + '/trt_sh/' #carpeta en donde se guardaran los .sh
ruta = '/home/rhernandez/trt_sh/' #carpeta en donde se guardaran los .sh
ruta_modelos = '/home/rhernandez/onnx_models/' #ruta de los modelos .onnx
ruta_modelos_trt = '/home/rhernandez/modelos_trt/' # ruta en donde se guardaran los modelos trt

try:
    os.mkdir(ruta)
except FileExistsError:
    print("Carpeta ya creada")  
    
try:
    os.mkdir(ruta_modelos_trt)
except FileExistsError:
    pass

modelos = ['casia', 'polyu', 'syntheticspvd', 'vera', 'iit', 'put', 'tongji']
tipos = ['fp32', 'fp16', 'int8']
batch_size_array = ['normal'] #['8', '16' ,'32', 'normal']

for modelo in modelos:
    ubicacion1 = ruta_modelos + modelo + '.onnx' #onnx
    ubicacion1_db = ruta_modelos + modelo + '_dynamic_batch.onnx'
    for batch_size in batch_size_array:
        
        for tipo in tipos:
            trt =  ruta + 'trt_quantization_' + tipo + '_' + modelo + '_' + batch_size + '.txt'


            ubicacion2 = ruta_modelos_trt + modelo + '_batch1_' + tipo + '.trt'
            linea1 = '/usr/src/tensorrt/bin/trtexec --onnx=' + ubicacion1 + ' --saveEngine=' + ubicacion2
            linea2 = '/usr/src/tensorrt/bin/trtexec --loadEngine=' + ubicacion2
           
            file = open(trt, "w")
            file.write(linea1 + os.linesep)
            file.write(linea2)

            if batch_size in ['8', '16', '32']:
                ubicacion2_db = ruta_modelos_trt + modelo + '_batch' + batch_size + '_' + tipo + '.trt'
                linea3 = '/usr/src/tensorrt/bin/trtexec --onnx=' + ubicacion1_db + ' --saveEngine=' + ubicacion2_db + ' --shapes=\'input\':' + batch_size + 'x128x128'
                linea4 = '/usr/src/tensorrt/bin/trtexec --loadEngine=' + ubicacion2_db
                file.write(os.linesep + linea3 + os.linesep )
                file.write(linea4)

            file.close()
            trtaux = trt.split('.')
            trt_sh = trtaux[0] + '.sh'
            os.rename(trt, trt_sh)
            comando = 'cd ' + ruta + ' && sh ' + 'trt_quantization_' + tipo + '_' + modelo + '_' + batch_size + '.sh'
            print(comando)
            os.system(comando)

