# Welcome to Process Prophet
Process prophet is a backend server that combines the power of the RMTPP model and process mining techniques. 
In particular, a process twin is created for an existing process that is able to: 
- simulate existing behavior.
- make predictions for ongoing process cases. 

This can provide important insights for project managers while improving decision making and resource allocation.

This model can be implemented in a variety of fields, as long as you have a timestamp, case id and an event name in your data. Therefore, Process Prophet also has a wide variety of possible areas of application. 

We also provide an `API` to agilize integration with other frontend interfaces. 

A CLI in the form of a `terminal user interface` is also provided for endusers that just want to use our provided functionalities. 



## Who is this documentation for? 
This documentation is intended to be used by frontend developers and data scientists that wish to integrate process prophet's backend functinality into their projects; or want to understand how we implement our models.

If you are an enduser of the CLI, check out the **user's guide**!


## API
**Intended for a frontend developer**: allows quick integration of our backend functionality by providing API endpoints that permit quick requests to our backend.  

## Backend
The core of **Process Prophet**: we support: 
- Preprocessing
- RMTPP model training
- Prediction generation 
- Process Mining 
- Conformance checking


## RMTPP
The RNN model we are using to generate our process twin. 


## CLI
We provide a CLI, here in the form of a terminal user interface, as one possible frontend for our backend. This was also done to agilize the testing of the backend. However, **Process Prophet's** potential can definitely stand out when implemented in your case specific software. We provide functionality and highly integrable software, so that endusers can incorporate our models to their own projects!