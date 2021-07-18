from decimal import Decimal as D


class Zscore:
    def __init__(self, skew, median, coff, measurement):
        self.skew = skew
        self.median = median
        self.coff = coff
        self.measurement = measurement

    def z_score_measurement(self):
        '''
         Z score
                  [y/M(t)]^L(t) - 1
           Zind =  -----------------
                      S(t)L(t)

                |       Zind            if |Zind| <= 3
                |
                |
                |       y - SD3pos
        Zind* = | 3 + ( ----------- )   if Zind > 3
                |         SD23pos
                |
                |
                |
                |        y - SD3neg
                | -3 + ( ----------- )  if Zind < -3
                |          SD23neg
        '''

        numerator = (self.measurement / self.median)**self.skew - D(1.0)
        denominator = self.skew * self.coff
        z_score = numerator / denominator

        def calc_stdev(sd):
            value = (1 + (self.skew * self.coff * sd))**(1 / self.skew)
            stdev = self.median * value
            return stdev

        if D(z_score) > D(3):
            SD2pos = calc_stdev(2)
            SD3pos = calc_stdev(3)

            SD23pos = SD3pos - SD2pos

            z_score = 3 + ((self.measurement - SD3pos) / SD23pos)

            z_score = float(z_score.quantize(D('0.01')))

        elif D(z_score) < -3:
            SD2neg = calc_stdev(-2)
            SD3neg = calc_stdev(-3)

            SD23neg = SD2neg - SD3neg

            z_score = -3 + ((self.measurement - SD3neg) / SD23neg)
            z_score = float(z_score.quantize(D('0.01')))

        else:
            z_score = float(z_score.quantize(D('0.01')))

        return z_score
