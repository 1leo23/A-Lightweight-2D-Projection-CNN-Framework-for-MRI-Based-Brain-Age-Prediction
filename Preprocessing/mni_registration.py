import ants

def mni_registration(input_path: str, output_path: str, template: ants.ANTsImage):
    """
    MNI 空間配準
    
    參數:
    - input_path: str, 輸入 NIfTI 檔案路徑
    - output_path: str, 輸出 NIfTI 檔案路徑
    - template: ants.ANTsImage, MNI 模板影像
    """
    brain_img = ants.image_read(input_path)
    reg = ants.registration(fixed=template, moving=brain_img, type_of_transform="SyN")
    warped = reg["warpedmovout"]
    ants.image_write(warped, output_path)