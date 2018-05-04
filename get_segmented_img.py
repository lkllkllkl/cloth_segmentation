from PIL import Image

classes = ['background', 'accessories', 'bag', 'belt', 'blazer', 
            'blouse', 'bodysuit', 'boots', 'bra', 'bracelet',
            'cape', 'cardigan', 'clogs', 'coat', 'dress', 
            'earrings', 'flats', 'glasses', 'gloves', 'hair', 
            'hat', 'heels', 'hoodie', 'intimate', 'jacket', 
            'jeans', 'jumper', 'leggings', 'loafers', 'necklace', 
            'panties', 'pants', 'pumps', 'purse', 'ring', 
            'romper', 'sandals', 'scarf', 'shirt', 'shoes', 
            'shorts', 'skin', 'skirt', 'sneakers','socks',
            'stockings', 'suit', 'sunglasses', 'sweater','sweatshirt',
            'swimwear', 't-shirt', 'tie', 'tights', 'top', 
            'vest', 'wallet', 'watch', 'wedges']


colormap = [[0, 0, 0], [165, 60, 45], [77, 164, 38], [202, 109, 103], [24, 18, 141],
            [37, 49, 58], [48, 36, 27], [187, 34, 238], [29, 128, 11], [109, 103, 226], 
            [19, 163, 16], [44, 203, 131], [222, 39, 93], [214, 118, 46], [35, 5, 110],
            [123, 156, 117], [46, 247, 79], [217, 184, 223], [30, 150, 240], [63, 111, 147], 
            [114, 213, 96], [31, 172, 214], [203, 131, 77], [25, 40, 115], [113, 191, 122], 
            [23, 251, 167], [68, 221, 17], [148, 196, 232], [73, 76, 142], [60, 45, 225],
            [157, 139, 253], [92, 239, 158], [52, 124, 178], [96, 72, 54], [190, 100, 160],
            [49, 58, 1], [32, 194, 188], [105, 15, 75], [254, 233, 26], [218, 206, 197], 
            [160, 205, 175], [238, 136, 187], [232, 4, 88], [185, 245, 35], [153, 51, 102], 
            [127, 244, 13], [124, 178, 91], [41, 137, 209], [253, 211, 52], [175, 25, 40],
            [229, 193, 166], [147, 174, 3], [84, 63, 111], [250, 145, 130], [215, 140, 20], 
            [20, 185, 245], [39, 93, 6], [174, 3, 66], [179, 113, 191]]


def getImages(origin_img, pred_img):
    """获取分割后的图片

    orgin_img: 原图
    pred_img: 预测结果
    return: 各分类分割图dict
    
    """
    origin_img = origin_img.convert('RGBA')
    pred_img = pred_img.convert('RGBA')

    class_imgs = {}
    width = origin_img.width
    height = origin_img.height
    
    for i in range(width):
        for j in range(height):
            pred_pixel = pred_img.getpixel((i, j))
            
            # 背景不作为单独类，直接跳过
            if pred_pixel == (0, 0, 0, 255) or not pred_pixel in colormap:
                continue

            if not pred_pixel in class_imgs.keys():
                class_img = Image.new('RGBA', (width, height), (0, 0 ,0, 0))    
                class_imgs[pred_pixel] = class_img
                print('class color(%s)'%(str(pred_pixel)))

            class_img = class_imgs[pred_pixel]
            class_img.putpixel((i, j), origin_img.getpixel((i, j)))

    return class_imgs


def main():
    origin_img = Image.open('val_1000_img.jpg')
    pred_img = Image.open('val_1000_annotation.jpg')
    class_imgs = getImages(origin_img, pred_img)
    for class_img in class_imgs.values():
        class_img.show()

if __name__ == '__main__':
    main()