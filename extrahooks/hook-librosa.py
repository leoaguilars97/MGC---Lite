from PyInstaller.utils.hooks import collect_data_files
print('Hey!!!')
datas = collect_data_files('librosa')
print('Obtenida data desde archivos LIBROSA!!! *************')
print(datas)