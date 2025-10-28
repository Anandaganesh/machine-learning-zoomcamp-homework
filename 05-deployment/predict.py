
churn = model.predict_proba(X)[0,1]

if churn >= 0.5:
    print('send email with promo')
else:
    print('dont do anything')