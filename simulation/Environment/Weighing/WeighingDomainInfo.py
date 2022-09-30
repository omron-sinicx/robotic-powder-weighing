

import numpy as np


class WeighingDomainInfo(object):
    """
    ゴールの粉体量は方策に入力するためにDRパラメータではなく観測にする
    粉体がなかなか落ちない場合があるのでspoon_frictionは実機だともう少し大きいことを想定した方がよさそう max_value=1. -> 1.2
    """

    def __init__(self, userDefinedSettings=None, domain_range=None, flag_list=None):
        if userDefinedSettings is None:
            print('domain info setting is passed !!')
            return
        self.userDefinedSettings = userDefinedSettings
        self.set_domain_parameter_all_space(domain_range=domain_range, randomization_flag=userDefinedSettings.DOMAIN_RANDOMIZATION_FLAG, flag_list=flag_list)

    def set_domain_parameter_all_space(self, domain_range=None, randomization_flag=True, flag_list=None):
        if randomization_flag:
            sampling_method = 'uniform'
        else:
            sampling_method = 'fix'

        if flag_list is not None:
            flag_list = np.where(flag_list, sampling_method, 'fix')
        else:
            flag_list = [sampling_method] * 8

        self.ball_radius = DomainParameter(name='ball_radius', initial_value=0.0050, min_value=0.0048, max_value=0.0052, sampling_method=flag_list[0])
        self.ball_mass = DomainParameter(name='ball_mass', initial_value=0.000030, min_value=0.000029, max_value=0.000031, sampling_method=flag_list[1])
        self.ball_friction = DomainParameter(name='ball_friction', initial_value=1., min_value=0.1, max_value=2., sampling_method=flag_list[2])
        self.ball_layer_num = DomainParameter(name='ball_layer_num', initial_value=15., min_value=15., max_value=20., sampling_method=flag_list[3])
        self.spoon_friction = DomainParameter(name='spoon_friction', initial_value=1., min_value=0.3, max_value=1.2, sampling_method=flag_list[4])
        self.goal_powder_amount = DomainParameter(name='goal_powder_amount', initial_value=0.005, min_value=0.004, max_value=0.016, sampling_method=flag_list[5])
        self.shake_speed_weight = DomainParameter(name='shake_speed_weight', initial_value=1., min_value=0.8, max_value=1.2, sampling_method=flag_list[6])
        self.gravity = DomainParameter(name='gravity', initial_value=-1., min_value=-1.1, max_value=-0.9, sampling_method=flag_list[7])

        # self.ball_radius = DomainParameter(name='ball_radius', initial_value=0.0050, min_value=0.0048, max_value=0.0052, sampling_method='fix')
        # self.ball_mass = DomainParameter(name='ball_mass', initial_value=0.000030, min_value=0.000029, max_value=0.000031, sampling_method='fix')
        # self.ball_friction = DomainParameter(name='ball_friction', initial_value=1., min_value=0.1, max_value=2., sampling_method='fix')
        # self.ball_layer_num = DomainParameter(name='ball_layer_num', initial_value=15., min_value=15., max_value=20., sampling_method='fix')
        # self.spoon_friction = DomainParameter(name='spoon_friction', initial_value=1., min_value=0.3, max_value=1.2, sampling_method='fix')
        # self.goal_powder_amount = DomainParameter(name='goal_powder_amount', initial_value=0.006, min_value=0.004, max_value=0.016, sampling_method='fix')
        # self.shake_speed_weight = DomainParameter(name='shake_speed_weight', initial_value=1., min_value=0.8, max_value=1.2, sampling_method='fix')
        # self.gravity = DomainParameter(name='gravity', initial_value=-1., min_value=-1.1, max_value=-0.9, sampling_method='fix')

        # ボールの数，摩擦，ゴール量
        # self.ball_radius = DomainParameter(name='ball_radius', initial_value=0.0050, min_value=0.0048, max_value=0.0052, sampling_method='fix')
        # self.ball_mass = DomainParameter(name='ball_mass', initial_value=0.000030, min_value=0.000029, max_value=0.000031, sampling_method='fix')
        # self.ball_friction = DomainParameter(name='ball_friction', initial_value=1., min_value=0.1, max_value=2., sampling_method='fix')
        # self.ball_layer_num = DomainParameter(name='ball_layer_num', initial_value=15., min_value=15., max_value=20., sampling_method='fix')
        # self.spoon_friction = DomainParameter(name='spoon_friction', initial_value=1., min_value=0.3, max_value=1.2, sampling_method='fix')
        # self.goal_powder_amount = DomainParameter(name='goal_powder_amount', initial_value=0.006, min_value=0.004, max_value=0.016, sampling_method='fix')
        # self.shake_speed_weight = DomainParameter(name='shake_speed_weight', initial_value=1., min_value=0.8, max_value=1.2, sampling_method='fix')
        # self.gravity = DomainParameter(name='gravity', initial_value=-1., min_value=-1.1, max_value=-0.9, sampling_method='fix')

    def set_parameters(self, set_info=None, type=None, reset_info=None):
        if reset_info is not None:
            set_info, type = reset_info
        target_domains = [self.ball_radius, self.ball_mass]
        other_domains = [self.ball_friction, self.ball_layer_num, self.spoon_friction, self.goal_powder_amount, self.shake_speed_weight, self.gravity]
        if type == 'set_split2':
            for domain, set_value in zip(target_domains, set_info):
                domain.set(set_value=set_value, set_method='rate_set')
            for domain in other_domains:
                domain.set()
        elif type == 'goal_condition':
            self.goal_powder_amount.set(set_value=set_info, set_method='direct_set')
            for domain in [self.ball_radius, self.ball_mass, self.ball_friction, self.ball_layer_num, self.spoon_friction, self.shake_speed_weight, self.gravity]:
                domain.set()
        else:
            for domain in target_domains:
                domain.set()
            for domain in other_domains:
                domain.set()

    def get_domain_parameters(self, type='all'):
        if type == 'all':
            target_parameters = [self.ball_radius, self.ball_mass, self.ball_friction, self.ball_layer_num, self.spoon_friction, self.goal_powder_amount, self.shake_speed_weight, self.gravity]
        elif type == 'observation':
            target_parameters = [self.ball_radius, self.ball_mass, self.ball_friction, self.ball_layer_num, self.spoon_friction, self.goal_powder_amount, self.shake_speed_weight, self.gravity]
        domain_parameters = []
        for parameter in target_parameters:
            domain_parameters.append(parameter.value)
        domain_parameters = np.array(domain_parameters)
        return domain_parameters

    def get_domain_parameter_dim(self):
        return len([self.ball_radius, self.ball_mass, self.ball_friction, self.ball_layer_num, self.spoon_friction, self.goal_powder_amount, self.shake_speed_weight, self.gravity])


class DomainParameter(object):
    def __init__(self, name, initial_value, min_value, max_value, sampling_method):
        assert (initial_value >= min_value) and (initial_value <= max_value), 'domain initial value is out of range'
        self.name = name
        self.value = initial_value
        self.min_value = min_value
        self.max_value = max_value
        self.min_range = 0.
        self.max_range = 1.
        self.sampling_method = sampling_method

    def set(self, set_value=None, set_range=None, set_method=None):
        if set_method == 'direct_set':
            self.value = set_value
        elif set_method == 'rate_set':
            self.value = (self.max_value - self.min_value) * set_value + self.min_value
        else:
            if self.sampling_method == 'uniform':
                if set_range is not None:
                    self.set_divided_space(set_range)
                origin_sample = np.random.rand()
                shifted_sample = (self.max_range - self.min_range) * origin_sample + self.min_range
                self.value = (self.max_value - self.min_value) * shifted_sample + self.min_value
            elif self.sampling_method == 'fix':
                pass
            elif self.sampling_method == 'set':
                self.value = set_value
            else:
                assert False, 'choose sampling method of the domain parameter'

        assert (self.value >= self.min_value) and (self.value <= self.max_value), 'domain value is out of range: {} < {} < {}'.format(self.min_value, self.value, self.max_value)

    def set_divided_space(self, domain_range):
        self.min_range = domain_range['min']
        self.max_range = domain_range['max']
