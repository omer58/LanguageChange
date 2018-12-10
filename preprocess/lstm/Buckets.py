class Buckets:

    def __init__(self, vals):
        self.w2yv_vals = vals
        self.buckets = self.create_buckets()

    def get_bucket_window_len(self, year):
        for b in self.buckets:
            if year in b:
                return len(b)

        return 5


    def create_buckets(self):
        max_year = self.w2yv_vals.shape[1]

        buckets = []
        prev = 0
        for year in range(0, 400):
            buckets.append(100)

        for year in range(400, 700):
            buckets.append(50)

        for year in range(700, 840):
            buckets.append(20)

        for year in range(850, 950, 10):
            buckets.append(range(prev,year))
            prev = year

        for year in range(960, max_year, 10):
            buckets.append(range(prev, year))
            prev = year

        #last bucket?


        return buckets

    def get_window(self, year):
        return 5 + (1018-year)/1018 * 100


    def is_in_bucket(self, prediction, target):

        window_len = self.get_window(target)

        return abs(target - prediction) < window_len:
        