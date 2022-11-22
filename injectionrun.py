class InjectionRun:
  def __init__(self, fn):
    import numpy as np
    import time as time
    import matplotlib.pyplot as plt
    import pandas as pd
    import bilby
    from bilby.core.prior import Uniform
    from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters, generate_all_bns_parameters
    ###############################################
    #
    # fn stores the filename of the dat file
    #
    ###############################################

    
    import pandas as pd
    df = pd.read_csv("/Ijnjection-Data/"+fn, sep=" ")
    self.mass_1 = df["mass_1"]
    self.mass_2 = df["mass_2"]
    self.a_1 = df["a_1"]
    self.a_2 = df["a_2"]
    self.tilt_1 = df["tilt_1"]
    self.tilt_2 = df["tilt_2"]
    self.phi_12 = df["phi_12"]
    self.phi_jl = df["phi_jl"]
    self.dec = df["dec"]
    self.ra = df["ra"]
    self.theta_jn = df["theta_jn"]
    self.psi = df["psi"]
    self.phase = df["phase"]
    self.geocent_time = df["geocent_time"]
    self.lambda_1 = df["lambda_1"]
    self.lambda_2 = df["lambda_2"]
    self.luminosity_distance = df["luminosity_distance"]
    
    
    #################################################
    # Data for GW170817

    # References:
    #   -> 
    #   -> https://github.com/mattpitkin/gw_notebooks/blob/master/EstimateDistance.ipynb

    #################################################
    
    #self.mass_1=1.46                  # In the future, use Uniform(1.46-0.10, 1.46+0.12)
    #self.mass_2=1.27                  # " " Uniform(1.27-0.09, 1.27+0.09)
    #self.chirp_mass = 1.186          # The chirp mass estimate is very well constrained (+- 0.001) but...
    #self.mass_ratio =                # Getting a good estimate of mass ratio is hard so I will use the two masses for now
    #self.a_1=0.264813928565
    #self.a_2=0.702414508316
    #self.tilt_1=2.58869030589
    #self.tilt_2=0.948965945788
    #self.phi_12= 6.04852924541
    #self.phi_jl=4.81306908412
    #self.luminosity_distance=40       # We want a better estimate of luminosity distance.
                                      # Current estimates are 26-48 Mpc.
    #self.theta_jn=2.74719229269       # There is some degeneracy between inlcination angle 
                                      # theta_jn and luminosity distance. However, we were
                                      # able to isolate the inclination angle due to the presence
                                      # of an EM counterpart.
    #self.psi=2.85798614041816         # In the future, get this from the strain data.
    #self.phase=2.371341               # Not sure how.
    #self.geocent_time=1187008882.43   # There is some time delay between the GW reaching H1 and 
                                      # L1 detectors. However it is small so I shall account for
                                      # this in future versions.
    #self.ra=3.44615261
    #self.dec= -0.408082219800
    #self.lambda_1 = 590.7203176488288 # 565.668993742142 -> 617.6943629446084 90% confidence interval
    #self.lambda_2 = 279.3840526017914 # 262.95917456078166 -> 292.1087835624852 90% confidence interval
                                      # Should actually either find the actual values
                                      # Or marginalise it over the whole set of values
                                      # Either way, it appears to have a small effect on the end result.

  #####################################################
  # To get the dictionary of injected values
  #####################################################
  def getDict(self):
    irdict = dict(
      mass_1=self.mass_1, mass_2=self.mass_2, a_1=self.a_1, a_2=self.a_2, tilt_1=self.tilt_1, tilt_2=self.tilt_2, 
      phi_12=self.phi_12, phi_jl=self.phi_jl, theta_jn=self.theta_jn, psi=self.psi,
      phase=self.phase, geocent_time=self.geocent_time, ra=self.ra, dec=self.dec, luminosity_distance = self.luminosity_distance,
      lambda_1=self.lambda_1, lambda_2=self.lambda_2)
    return(irdict)
  
    
  ###################################################
  # 
  ###################################################
  def setLims(dl_min=26, dl_max=48):
    self.dl_min = dl_min
    self.dl_max = dl_max
  


  ####################################################
  # Calculates results, stores them in savedir directory.
  # Returns result for luminosity distance
  # Takes input nl for nlive. Larger nlive means better
  # estimate but takes longer to run. nl = 5 took 45 mins
  # Model = IMRPhenomPV2 / IMRPhenomPv2_NRTidal / SpinTaylorT4
  ####################################################

  def estimate(self, model, nl, savedir, signal_duration=1):
    import numpy as np
    import matplotlib.pyplot as plt
    import bilby
    from bilby.core.prior import Uniform
    from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters, generate_all_bns_parameters
    
    self.nl = nl
    self.model = model
    
    np.random.seed(1234) # For reproducibility
    injection_parameters = self.getDict()

    waveform_arguments = dict(waveform_approximant=model, reference_frequency=50., minimum_frequency=20., catch_waveform_errors=True)
    sampling_frequency = 2048.
    
    try:
      duration = signal_duration
      waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
                        waveform_arguments=waveform_arguments)
      ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
      ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
                start_time=injection_parameters['geocent_time'] - 3)
      injection = ifos.inject_signal(
            waveform_generator=waveform_generator,
                parameters=injection_parameters)

    except ValueError as ve:
      duration = math.ceil(8*float(str(ve).split()[8][0:-2]))/8
      waveform_generator = bilby.gw.WaveformGenerator(
            duration=duration, sampling_frequency=sampling_frequency,
                frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
                    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_neutron_star_parameters,
                        waveform_arguments=waveform_arguments)
      ifos = bilby.gw.detector.InterferometerList(['H1', 'L1'])
      ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=sampling_frequency, duration=duration,
                start_time=injection_parameters['geocent_time'] - 3)
      injection = ifos.inject_signal(
            waveform_generator=waveform_generator,
                parameters=injection_parameters)

    H1 = ifos[0]
    H1_injection = injection[0]

    prior = bilby.core.prior.PriorDict()

    prior['luminosity_distance'] = Uniform(name='luminosity_distance', minimum=26,maximum=48)

    for key in ["mass_1", "mass_2", 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'ra', 'dec', 'theta_jn', 'geocent_time', 'lambda_1', 'lambda_2']:
        prior[key] = injection_parameters[key]
    prior

    prior['phase'] = Uniform(1.87392, 2.887066, name='phase')                 # In future versions, run it across the whole 
    prior['psi'] = Uniform(2.8464217993353, 2.866418390430222, name = 'psi')  # psi and phase. But this gives bimodal graph
                                                                              # There is some bimodality in phase
                                                                              # There is a way to extract psi and phase but
                                                                              # I have yet to learn that. 

    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            interferometers=ifos, waveform_generator=waveform_generator, priors=prior,
                time_marginalization=False, phase_marginalization=False, distance_marginalization=False)

    result_short = bilby.run_sampler(
            likelihood, prior, sampler='dynesty', outdir=savedir+today.strftime("%Y-%m-%d-%H-%M-%S")+"/result"+, label="GW170817",
                conversion_function=bilby.gw.conversion.generate_all_bns_parameters,
                    nlive=nl, dlogz=3,  # Remove arguments in future versions (or on a better computer)
                        clean=True
                        )
    
    file = open("runpar.txt", "x")
    file.writelines = (["Run Characteristics:", # Saves the parameters, model, number of samples and np.random seed
                        "mass_1: " + mass_1,
                        "mass_2: " + mass_2,
                        "a_1: " + a_1,
                        "a_2: " + a_2,
                        "tilt_1: " + tilt_1,
                        "tilt_2: " + tilt_2,
                        "phi_12: " + phi_12,
                        "phi_jl: " + phi_jl,
                        "lambda_1: " + lambda_1,
                        "lambda_2: " + lambda_2,
                        "theta_jn" + theta_jn,
                        "dec: " + dec,
                        "ra: " + ra,
                        "geocent_time: " + geocent_time,
                        "model: " + model,
                        "nl: " + nl,
                        "seed: " + str(np.random.get_state())
                       ])
    
    result_short.posterior
    result_short.posterior["luminosity_distance"]
    Mc = result_short.posterior["luminosity_distance"].values

    lower_bound = np.quantile(Mc, 0.05)
    upper_bound = np.quantile(Mc, 0.95)
    median = np.quantile(Mc, 0.5)
    print("Luminosity Distance = {} with a 90% C.I = {} -> {}".format(median, lower_bound, upper_bound))
    result_short.priors
    result_short.sampler_kwargs["nlive"]
    
    self.result =  result_short
  
  def getResult():
    return self.result
