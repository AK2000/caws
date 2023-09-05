import caws

def load_endpoint(config_obj, name):
    args_dict = config_obj["endpoints"][name]
    return caws.Endpoint(name, 
                         **args_dict,
                         monitoring_avail=True,
                         monitor_url=config_obj["caws_monitoring_db"])