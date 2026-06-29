$Python = "C:\Users\kaush\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $Python scripts\create_sample_docs.py
& $Python training\build_dataset.py
& $Python training\train_model.py
