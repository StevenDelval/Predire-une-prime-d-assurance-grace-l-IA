def cat_bmi(bmi):
    """
    Revoie la categorie dans laquelle le bmi ce situe
    """
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25:
        return "healthy"
    elif bmi <30:
        return "overweight"
    elif bmi < 40:
        return "obesity"
    else:
        return "morbid_obesity"