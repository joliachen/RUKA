from ruka_hand.learning.preprocessor import Preprocessor

save_dirs = ["/home/jolia/vr-hand-tracking/Franka-Teach/RUKA/data/right_hand/demonstration_test"]
preprocessor = Preprocessor(
    save_dirs=save_dirs,
    frequency=-1,
    module_keys=["manus", "ruka"],
    # visualize=False,
)

processes = preprocessor.get_processes()
for process in processes:
    process.start()

for process in processes:
    process.join()
