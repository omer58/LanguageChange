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
        for year in range(0, 401, 100):
            buckets.append(range(prev,year))
            prev = year

        for year in range(450, 701, 50):
            buckets.append(range(prev,year))
            prev = year

        for year in range(720, 840, 20):
            buckets.append(range(prev,year))
            prev = year

        for year in range(850, 950, 10):
            buckets.append(range(prev,year))
            prev = year

        for year in range(960, max_year, 10):
            buckets.append(range(prev, year))
            prev = year

        #last bucket?


        return buckets



    def is_in_bucket(self, prediction, target):
        for b in self.buckets:
            if target in b:
                window_len = len(b)

                if abs(target - prediction) < window_len:
                    return True

        return False
