
class Robyn(object):

    def __init__(self):
        self.set_country = 'DE'
        # include all the dictionary items meant for global parameters, this requires good documentation
        self.check_conditions() #maybe?

    def check_conditions(self):
        pass

    def input_wrangling(self, df): # where should we put this??
        check_conditions()
        pass

    def get_hypernames(self):
        pass

    @staticmethod
    def michaelis_menten(spend, vmax, km):
        pass

    @staticmethod
    def adstockGeometric(x, theta):
        pass

    @staticmethod
    def helperWeibull(x, y, vec_cum, n):
        pass

    @staticmethod
    def adstockWeibull(x, shape, scale):
        pass

    @staticmethod
    def transformation(x, adstock, theta=None, shape=None, scale=None, alpha=None, gamma=None, stage=3):
        pass

    @staticmethod
    def unit_format(x_in):
        pass

    @staticmethod
    def rsq(true, predicted):
        pass

    def lambdaRidge(x, y, seq_len=100, lambda_min_ratio=0.0001):
        pass

    def decomp(coefs, dt_modAdstocked, x, y_pred, i, d):
        pass

    def refit(x_train, y_train, lambda_: int, lower_limits: list, upper_limits: list):
        pass

    def fit(self): #this replace the original mmm + Robyn functions
        pass

    def budget_allocator(self,model_id): #this is the last step_model allocation
        pass




########### Demo

df = read.csv(xxxx)


mmm = Robyn(set_country="DE")

df_transformed = mmm.input_wrangling(df)

mmm.fit(df_transformed)

# after selecting model
mmm.allocate_budget(modID = "3_10_2", scenario = "max_historical_response",
                    channel_constr_low = c(0.7, 0.75, 0.60, 0.8, 0.65),
                    channel_constr_up = c(1.2, 1.5, 1.5, 2, 1.5))