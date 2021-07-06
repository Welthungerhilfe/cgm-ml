from decimal import Decimal as D


class Value:
    def __init__(self, chart, weight, muac,
                 age_in_days, sex, height):
        self.chart = chart
        self.weight = weight
        self.muac = muac
        self.age_in_days = age_in_days
        self.sex = sex
        self.height = height

    def resolve_table(self):

        if self.chart == 'wfl' and D(self.height) > 110:
            table_chart = 'wfh'
            table_age = '2_5'
        elif self.chart == 'wfh' and D(self.height) < 65:
            table_chart = 'wfl'
            table_age = '0_2'
        else:
            table_chart = self.chart
            if self.chart == 'wfl':
                table_age = '0_2'
            if self.chart == 'wfh':
                table_age = '2_5'

        if self.sex == 'M':
            table_sex = 'boys'
        elif self.sex == 'F':
            table_sex = 'girls'

        if self.chart in ["wfa", "lhfa"]:
            table_age = "0_5"
            table_chart = self.chart

        table = "%(table_chart)s_%(table_sex)s_%(table_age)s" %\
                {"table_chart": table_chart,
                 "table_sex": table_sex,
                 "table_age": table_age}
        return table

    def get_values(self, table_name, growth):
        table = growth[table_name]
        if self.chart in ["wfh", "wfl"]:
            if D(self.height) < 45:
                raise Exception("too short")
            if D(self.height) > 120:
                raise TypeError("too tall")
            closest_height = float("{0:.1f}". format(D(self.height)))
            scores = table.get(str(closest_height))
            if scores is not None:
                return scores
            raise TypeError(
                "SCORES NOT FOUND FOR HEIGHT :%s", (self.closest_height))

        if self.chart in ['wfa', 'lhfa']:
            scores = table.get(str(self.age_in_days))
            if scores is not None:
                return scores
            raise TypeError(
                "SCORES NOT FOUND BY DAY : %s", self.age_in_days)

    def resolve_value(self, value):
        skew = D(value.get("L"))
        median = D(value.get("M"))
        coff = D(value.get("S"))
        if self.chart == 'wfa' or self.chart == 'wfl' or self.chart == 'wfh':
            y = D(self.weight)
        elif self.chart == 'lhfa':
            y = D(self.height)
        return skew, median, coff, y
