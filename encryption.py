try:
    path = input(r'Enter path of image :')
    key = int(input('Enter key for encryption of image : '))

    print('The path of file:', path)
    print("Key for ecryption:",key)

    fin = open(path, 'rb')
    image = fin.read()
    fin.close()
    image = bytearray(image)
    
    for index, values in enumerate(image):
        image[index] = values ^ key
    
    fin = open(path, 'wb')
    fin.write(image)
    fin.close()
    print('Enccryption Done.....')
except Exception:
    print('Error caught :',Exception.name)    
