"""
One-time setup: saves your IBM Quantum account on this computer (nothing is written to the repo).
Run from the repo root:  python3 collect/setup_account.py
It asks for your API key (typing is hidden) and your instance CRN, then tests the connection.
"""
from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService

key = getpass("Paste your IBM Quantum API key and press Enter: ").strip()
crn = input("Paste your instance CRN (crn:v1:bluemix:...) and press Enter: ").strip()

QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token=key,
    instance=crn,
    set_as_default=True,
    overwrite=True,
)
print("Account saved. Testing connection...")

service = QiskitRuntimeService()
names = [b.name for b in service.backends()]
print("It works! Backends you can see:", names)
