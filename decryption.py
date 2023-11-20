try:
    path = input(r'Enter path of image:')

    key = int(input('Enter key for encryption of Image :'))
    print('The path of file :', path)
    print('Note : Encryption key and Decryption key must be same.')
    print('Key for Decryption :', key)

    fin = open(path, 'rb')
    image = fin.read()
    fin.close()

    image = bytearray(image)

    for index, values in enumerate(image):
        image[index] = values ^ key
    
    fin = open(path, 'wb')

    fin.write(image)
    fin.close()
    print('Decryption Done....')

except Exception:
    print('Error caught : ', Exception.name)

