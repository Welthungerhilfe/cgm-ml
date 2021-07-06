from decimal import Decimal as D


class Zscore:
    def __init__(self, skew, median, coff, measurement):
        self.skew = skew
        self.median = median
        self.coff = coff
        self.measurement = measurement

    def z_score_measurement(self):
        numerator = (self.measurement / self.median)**self.skew - D(1.0)
        denominator = self.skew * self.coff
        zScore = numerator / denominator

        def calc_stdev(sd):
            value = (1 + (self.skew * self.coff * sd))**(1 / self.skew)
            stdev = self.median * value
            return stdev

        if D(zScore) > D(3):
            SD2pos = calc_stdev(2)
            SD3pos = calc_stdev(3)

            SD23pos = SD3pos - SD2pos

            zScore = 3 + ((self.measurement - SD3pos) / SD23pos)

            zScore = float(zScore.quantize(D('0.01')))

        elif D(zScore) < -3:
            SD2neg = calc_stdev(-2)
            SD3neg = calc_stdev(-3)

            SD23neg = SD2neg - SD3neg

            zScore = -3 + ((self.measurement - SD3neg) / SD23neg)
            zScore = float(zScore.quantize(D('0.01')))

        else:
            zScore = float(zScore.quantize(D('0.01')))

        return zScore
