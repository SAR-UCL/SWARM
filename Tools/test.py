
def maxwell():
    from scipy.stats import maxwell
    import matplotlib.pyplot as plt
    #import seaborn as sns


    r = maxwell.rvs(size=1000)
    fig, ax = plt.subplots(1, 1)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)

    plt.show()

#print(r)

import plotly.express as px

fig = px.scatter(new_profile[:100],
          x='followers',
          y='total_stars',
          color='forks',
          size='contribution')
fig.show()