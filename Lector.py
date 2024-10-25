import cv2
import zxingcpp # type: ignore
import numpy as np

def preprocess_image(image):
    """Preprocess the image to improve barcode detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def parse_cedula_data(text):
    """Parse the barcode text for Costa Rican ID card data."""
    try:
        cedula = text[:9]
        nombre = text[9:35].strip()
        apellido1 = text[35:61].strip()
        apellido2 = text[61:87].strip()
        sexo = "Masculino" if text[87] == "1" else "Femenino"
        fecha_nacimiento = f"{text[94:96]}/{text[92:94]}/{text[88:92]}"
        fecha_vencimiento = f"{text[102:104]}/{text[100:102]}/{text[96:100]}"
        
        return {
            "Cédula": cedula,
            "Nombre": nombre,
            "Primer Apellido": apellido1,
            "Segundo Apellido": apellido2,
            "Sexo": sexo,
            "Fecha de Nacimiento": fecha_nacimiento,
            "Fecha de Vencimiento": fecha_vencimiento
        }
    except IndexError:
        return None

def read_cedula_barcode(image_path):
    """Read and parse the barcode from a Costa Rican ID card image."""
    try:
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read the image. Please check the file path.")

        # Preprocess the image
        processed_img = preprocess_image(img)

        # Read barcodes
        results = zxingcpp.read_barcodes(processed_img)

        if len(results) == 0:
            print("Could not find any barcode. Please ensure the image contains a clear barcode.")
            return

        for result in results:
            print(f"Found barcode:")
            print(f"Format: {result.format}")
            print(f"Position: {result.position}")
            
            # Parse the cedula data
            cedula_data = parse_cedula_data(result.text)
            if cedula_data:
                print("\nCédula Information:")
                for key, value in cedula_data.items():
                    print(f"{key}: {value}")
            else:
                print("\nRaw barcode text:")
                print(result.text)
                print("\nWarning: Could not parse the barcode data as a Costa Rican ID. The format may be different than expected.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage
image_path = r"C:\Users\shann\Pictures\img2.jpg"
read_cedula_barcode(image_path)