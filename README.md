# PoisonHub – Framework for Backdoor Attacks and Defenses

ML Security App is a web-based framework for analyzing backdoor attacks and defense mechanisms in neural networks. It provides a unified experimental environment in which users can set up machine learning pipelines, apply different backdoor attacks and defenses, and evaluate their impact on model behavior and performance.

The project is intended to simplify experimentation in the area of machine learning security. By combining a Python-based backend with an interactive graphical interface, the application allows experiments to be configured and executed without modifying the underlying source code. Users can select datasets and models, adjust training settings, and define attack or defense parameters directly through the interface.

The system is designed to be easy to extend and consistent in execution. New attacks, defenses, datasets, or models can be added by implementing a small set of well-defined interfaces, while all experiments follow the same execution flow. This ensures that training, evaluation, and result reporting remain comparable across different setups, making the framework suitable for structured experimentation and comparison of backdoor attack and defense techniques.

---

## High-Level Overview

The system consists of two main components:

- **Backend**, implemented in Python, which contains all core logic, machine learning models, attacks, defenses, and evaluation code  
- **Frontend**, implemented as a web application, which provides a user-friendly interface for configuring experiments and inspecting results  

---

## Backend Overview

The backend is responsible for running all machine learning–related parts of the framework. It handles dataset loading, model training, execution of backdoor attacks, applying defenses, evaluating results, and generating outputs for visualization. All backend components are implemented in Python and organized into separate modules, each covering a specific part of the experiment process.

Each experiment follows a clear and consistent sequence of steps. Data is first prepared and loaded, after which an optional backdoor attack can be applied. The model is then trained and evaluated, followed by an optional defense step and final result calculation. This structure ensures that experiments are executed in the same way across different configurations, making results easier to compare.

The backend is organized to allow new functionality to be added when needed. New attacks, defenses, datasets, or models can be introduced by implementing the appropriate interface and placing the implementation in the corresponding module. Once added, these components can be used in experiments alongside existing ones without changes to the rest of the system.

---

## Frontend Overview

The frontend is a React-based web application built using **Next.js** and **TypeScript**.  
It serves as the primary interface for interacting with the framework and was designed to make complex backdoor experiments intuitive and easy to configure.

### User Workflow

1. **Dataset and Model Selection**  
   The user selects a dataset (e.g. MNIST or CIFAR-10) and a model family. After choosing a model, training hyperparameters such as learning rate, number of epochs, momentum, number of runs, and random seeds can be configured.

2. **Attack and Defense Configuration**  
   The user can select a backdoor attack and, optionally, a defense.  
   Once selected, only the parameters specific to the chosen attack or defense are shown in the interface. Each parameter is clearly explained, allowing users to understand its purpose and adjust the configuration accordingly.

3. **Overview**  
   Before running the experiment, the application displays a summary of all selected components and parameters in one place.

4. **Execution and Results**  
   After starting the experiment, the frontend communicates with the backend and waits for completion.  
   Once finished, the results page displays:
   - clean accuracy  
   - attack success rate (ASR)  
   - post-defense accuracy and ASR  
   - relative improvements  
   - visual examples of poisoned inputs and triggers (for image-based attacks)

---

## Technologies Used

### Backend

- Python  
- Flask  
- flask-cors  
- PyTorch  
- torchvision  
- adversarial-robustness-toolbox (ART)  
- NumPy  
- Pillow  
- Numba  
- Pandas  
- Packaging  

### Frontend

- TypeScript  
- React  
- Next.js  
- CSS / utility-based UI components  

---

## Implemented Attacks and Defenses

The framework includes a wide range of backdoor attacks and defenses, covering both commonly used baseline methods and more advanced research approaches. All listed methods are fully integrated into the framework and can be configured and executed through the graphical interface.

### Implemented Backdoor Attacks

- BadNets  
- Blend  
- AdaptiveBlend  
- BPP (Backdoor Poisoning with Perturbations)  
- SIG  
- WaNet  
- LabelConsistent  
- LIRA  
- AdvDoor  
- Narcissus  
- IAD (Input-Aware Dynamic Attack)  
- DFST (Deep Feature Space Trojan)  
- Grond  

For reference and comparison, a **NoAttack** option is also provided.

---

### Implemented Defenses

- FinePruning  
- ANP (Adversarial Neuron Pruning)  
- IBAU  
- NAD (Neural Attention Distillation)  
- ABL (Anti-Backdoor Learning)  
- Neural Cleanse  
- Spectral Signatures  
- STRIP  
- BAN  
- CBD  

A **NoDefense** option is included to evaluate attack effectiveness without applying any mitigation.

---

## Metrics

The framework supports multiple evaluation metrics, including:

- **AccuracyDifference** – clean accuracy and ASR change  
- **AttackIntensity** – attack effectiveness measure  
- **CalculateStd** – mean and standard deviation over multiple runs  

---

## Running the Project

### Frontend Setup

To run the frontend, first navigate to the frontend directory:

```bash
cd ml-security-frontend
```

Install the required Node.js dependencies:

```bash
npm install
```

After the installation is complete, start the server:

```bash
npm run dev
```

### Backend Setup

To run the backend, first navigate to the backend directory:

```bash
cd ml-security-backend
```

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

After the installation is complete, start the backend server:

```bash
python app.py
```
