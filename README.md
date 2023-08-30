# Clickstream Analytics Project Overview

## Challenge
The challenge this project aimed to tackle was predicting purchase intent of customers on an ecommerce platform. The goal was to understand the customer's purchasing behavior and likelihood of buying a product based on their engagement with the site.

Ecommerce platforms often involve complex customer interactions where users browse, compare products, leave, return, add items to the cart, and may even abandon carts before making a purchase decision. Understanding these interactions and predicting purchase propensity can help improve product recommendations, boost sales, and enhance the overall customer experience. 

## Development and Solution

### Approach:

1. **Data Processing (Batch)**: The first step was to process historical clickstream data accumulated from the ecommerce site. This data, containing granular details about various user activities, was transformed using batch processing to generate meaningful features for the model. 

2. **Medallion Architecture**: The project made use of a medallion architecture, dividing the data processing workflow into three levels - Bronze, Silver, and Gold. 

    - **Bronze layer**: Raw, unaltered data is stored as it is ingested into Databricks, without any structural modifications.
  
    - **Silver layer**: Here, data from the Bronze layer is transformed for technical accessibility but without business-level interpretation. This stage served as the main starting point for diverting data into different streams for Data Scientists and Data Engineers.
    
    - **Gold layer**: In this final level, five sets of metrics or features are derived from the processed event data to assist with model training. These features represent comprehensive knowledge about the state of each user, product, and shopping cart.

3. **Model Training**: Leveraged the historical backlog data to train a model that can estimate propensities from the event data of the electronics ecommerce site.

<img src='https://brysmiwasb.blob.core.windows.net/demos/images/cs_part_1.png' width=750>

In terms of tools, Databricks was the primary platform used for data processing and model training, with the organization and storage of data being handled via DBFS. The coding for data transformation and modeling was carried out using Python and PySpark, the ML model using MLFLOW and the Orchestration using Databricks Jobs.

## Impact of the work
The solutions developed in this project aimed to tackle two key challenges:

1. **Improve Purchase Conversion Rate**: By accurately estimating the probability of a customer making a purchase, the ecommerce platform can provide more personalized product suggestions that align with the customer's preferences and needs, thereby increasing conversion rates.

2. **Enhance Customer Experience**: With better understanding of the customer's purchase intent, the ecommerce platform can optimize their marketing strategies. For instance, discounts or promotions could be leveraged for customers with moderate purchase propensity, whereas customers showing high purchase likelihood might be shown complementary products to maximize sale value.

## Tech Stack

The technology stack involved in this project included:

- Databricks: For data processing and model training.
- Python & PySpark: For writing scripts for transformations, aggregations and model generation.