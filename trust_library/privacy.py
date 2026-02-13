from ml_privacy_meter import MembershipInferenceAttack

# 1) Entrena tu modelo (cualquier ML) EPISILON*
model = ...

# 2) Ejecuta ataque de inferencia de membresía con ml_privacy_meter
mia = MembershipInferenceAttack(...)
results = mia.attack(model, data_member, data_nonmember)

# 3) Obtén TPR/FPR
tpr = results.true_positive_rate
fpr = results.false_positive_rate

# 4) Implementa cálculo de Epsilon* (según artículo)
def compute_epsilon_star(tpr, fpr):
    # fórmula inspirada en la definición de privacidad diferencial
    # (típicamente: log(tpr/fpr) / ? según sección del paper)
    return np.log(tpr / fpr)

eps_star = compute_epsilon_star(tpr, fpr)


