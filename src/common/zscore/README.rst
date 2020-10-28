Z score
=======

This libray is Used for measuring Z score of Children (0-5 Years) based on standard provided by WHO 2006

.. image:: https://img.shields.io/pypi/v/cgmzscore.svg
    :target: https://pypi.org/project/cgmzscore/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/wheel/cgmzscore.svg
    :target: https://pypi.org/project/cgmzscore/

.. image:: https://img.shields.io/pypi/pyversions/cgmzscore.svg
    :target: https://pypi.org/project/cgmzscore/

REQUIREMENTS
============

* Python 2.7.x, Python 3.x or later

INSTALLATION
============
`pip install cgmzscore`

EXAMPLE USAGE
=============

calculate z score for weight vs age

.. code-block:: python

    from cgmzscore import Calculator

    v = Calculator().zScore_wfa(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

calculate z score for weight vs length/height

.. code-block:: python

    from cgmzscore import Calculator

    v = Calculator().zScore_wfl(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

calculate z score for weight vs length/height and both wfl and wfh works same

.. code-block:: python

    from cgmzscore import Calculator

    v = Calculator().zScore_wfh(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

calculate z score for length vs age

.. code-block:: python

    from cgmzscore import Calculator

    v = Calculator().zScore_lhfa(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

calculate all three z score

.. code-block:: python

    from cgmzscore import Calculator

    v = Calculator().zScore(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

calculate all three z score along with class

.. code-block:: python

    from cgmzscore import Calculator

    v = calculator.zScore_withclass(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

find child is SAM/MAM/Healthy

.. code-block:: python

    from cgmzscore import Calculator

    v = Calculator().SAM_MAM(weight="7.853",muac="13.5",age_in_days='16',sex='M',height='73')

Chart for z score for weight vs age

.. code-block:: python

    from cgmzscore import Chart

    Chart().zScore_wfa_chart(weight=[7.853],muac=[13.5],age_in_days=[160],sex='M',height=[73]).show()

Chart for z score for length vs age

.. code-block:: python

    from cgmzscore import Chart

    Chart().zScore_lhfa_chart(weight=[7.853],muac=[13.5],age_in_days=[160],sex='M',height=[73]).show()

Chart for z score for weight vs length

.. code-block:: python

    from cgmzscore import Chart

    Chart().zScore_wfh_full_chart(weight=[7.853],muac=[13.5],age_in_days=[160],sex='M',height=[73]).show()
