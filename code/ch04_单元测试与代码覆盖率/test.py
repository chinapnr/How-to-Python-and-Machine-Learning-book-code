import unittest
from scale import Scale

# 创建单元测试类，并在 setUp 和 tearDown 中对 scale 属性进行初始化和还原。

class AnyScaleTest(unittest.TestCase):
  
    def setUp(self):
        self.scale = Scale()   

    def tearDown(self):
        self.scale = None    

    # 测试用例 1，测试十进制下的 16 转化为十六进制。

    def test1_10_16_16(self):
        self.assertEquals('10', self.scale.any_scale(10, 16, 16))

    # 测试用例 2，测试十进制下的 17 转化为八进制。

    def test2_10_17_8(self):
        self.assertEquals('19', self.scale.any_scale(10, 17, 8))

    # 测试用例 3，测试八进制下的 19 转化为十进制。

    def test3_8_19_10(self):
        self.assertEquals('17', self.scale.any_scale(8, 19, 10))


if __name__ == '__main__':
    unittest.main() 
