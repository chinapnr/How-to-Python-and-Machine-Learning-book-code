# 这段代码中，我们实现了一个 n 进制转换的小工具，可以在任意进制之间进行转化。

class Scale(object):

    #这里用常用的字母代表数字的字典。

    dic = {'10': 'A', '11': 'B', '12': 'C ', 
           '13': 'D', '14': 'E', '15': 'F'}

    # 定义一个函数将 weight 进制的某一位的值对应的十进制的值算出来。

    @staticmethod
    def place_value(n_value, scale, digits):
        # 某一位的权值,初始为 1
        weight = 1
        for i in range(1, digits + 1):
            weight = scale * weight
        return n_value * weight

    # 定义一个函数将 scale 进制的值 value 转为对应十进制的值。

    @staticmethod
    def n_2_decimal(value_, scale):
        sum_ = 0
        n = len(str(value_))
        for i in range(1, n + 1):
            sum_ = sum_ + Scale.place_value(int(str(value_)
                                                  [i-1]), scale, n-i) 
        return sum_

    # 这个函数将十进制的值 value 转为对应 scale 进制的值。

    @staticmethod
    def decimal_2_n(value_, scale):
        arr = []
        i = 0
        while value_ is not 0:
            rem = value_ % scale
            if rem >= 16:
                rem = "*" + str(rem) + "*"
            elif 10 <= rem <= 15:
                rem = Scale.dic[str(rem)]
            value_ = value_ // scale
            arr.append(rem)
            i += 1
        return arr

    # 最后，这个函数可以进行不同进制间的转化。

    @staticmethod
    def any_scale(scale1_, value_, scale2_):
        mid_value = Scale.n_2_decimal(value_, scale1_)
        fin_value = Scale.decimal_2_n(mid_value, scale2_)
        fin_value.reverse()
        fin_value = ''.join([str(x) for x in fin_value])
        return fin_value
