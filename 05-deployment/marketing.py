

print(f'prob of churning = {churn}')

if churn >= 0.5:
    print('send email with promo')
else:
    print('dont do anything')