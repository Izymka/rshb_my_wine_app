import os

# faiss-cpu и torch на macOS каждый тянут свою копию OpenMP, и вторая инициализация роняет
# процесс с OMP: Error #15. Флаг разрешает загрузить обе. Ставим до импорта torch и faiss,
# то есть здесь — этот модуль выполняется раньше всех остальных в пакете.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
