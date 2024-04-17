from zigzag.api import get_hardware_performance_zigzag

opt = 'EDP'
model = "alexnet"
onnx_model_path = f"zigzag/inputs/examples/workload/{model}.onnx"
precision = f"zigzag/inputs/examples/workload/{model}.json"
workload = onnx_model_path

hwarch = "Edge_TPU_like"
mapping = f"zigzag.inputs.examples.mapping.default"
accelerator = f"zigzag.inputs.examples.hardware.{hwarch}"

dump_filename_pattern=f"outputs/{hwarch}-{model}-layer_?.json"
pickle_filename = f"outputs/{hwarch}-{model}-saved_list_of_cmes.pickle"

energy, latency, cme = get_hardware_performance_zigzag(workload=workload,
                                                       precision=precision,
                                                       accelerator=accelerator,
                                                       mapping=mapping,
                                                       opt=opt,
                                                       dump_filename_pattern=dump_filename_pattern,
                                                       pickle_filename=pickle_filename)
print(f"Total network energy = {energy:.2e} pJ")
print(f"Total network latency = {latency:.2e} cycles")
print(f"Total edp = {energy*latency:.2e} pJ*cycles")